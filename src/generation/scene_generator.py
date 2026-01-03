# src/generation/scene_generator.py
from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional

from src.generation.prompt_builder import (
    build_prompt_bundle,
    build_repair_prompt_bundle,
)
from src.generation.retry import build_retry_prompt_bundle


class SceneGenerator:
    """
    Research-grade scene generation orchestrator.

    Responsibilities:
    - build the initial prompt bundle from continuity package
    - run backend generation
    - score output using scorer
    - manage retry loop
    - switch from normal generation to repair generation when needed
    - attach rich metadata needed by continuity manager and later evaluation

    This class assumes:
    - backend.generate(prompt_bundle) -> Dict[str, Any]
    - optional backend.generate_repair(prompt_bundle) -> Dict[str, Any]
    - scorer.score(scene_packet, generation_result, prompt_bundle, continuity_package) -> Dict[str, Any]
      OR scorer.score(...) with fewer args (handled safely)

    New support:
    - continuity_package["prompt_bundle_override"]
      lets repair engine inject a precomputed prompt/control/reference bundle.
    """

    def __init__(
        self,
        backend: Any,
        scorer: Optional[Any] = None,
        repair_backend: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.backend = backend
        self.scorer = scorer
        self.repair_backend = repair_backend
        self.config = dict(config or {})

        generation_cfg = self.config.get("generation", {})
        retry_cfg = self.config.get("retry", {})
        repair_cfg = self.config.get("repair", {})

        self.max_retries = int(
            retry_cfg.get(
                "max_retries",
                generation_cfg.get("max_retries", 2),
            )
        )
        self.repair_after_retry = bool(
            repair_cfg.get(
                "repair_after_retry",
                generation_cfg.get("repair_after_retry", True),
            )
        )
        self.accept_score_threshold = float(
            generation_cfg.get("accept_score_threshold", 0.60)
        )
        self.hard_fail_on_identity = bool(
            generation_cfg.get("hard_fail_on_identity", False)
        )
        self.hard_fail_on_location = bool(
            generation_cfg.get("hard_fail_on_location", False)
        )
        self.strict_repair_mode = bool(
            repair_cfg.get("strict_repair_mode", True)
        )

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------

    def generate_scene(self, continuity_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point used by pipeline.

        Returns a result dictionary containing:
        - ok
        - accepted
        - generation_result
        - score_report
        - drift_report
        - prompt_bundle
        - retry_count
        - backend_used
        - attempts
        - repair_used
        """
        continuity_package = self._safe_dict(continuity_package)
        scene_id = self._scene_id(continuity_package)
        scene_packet = self._safe_dict(continuity_package.get("scene_packet", {}))

        base_prompt_bundle = self._resolve_base_prompt_bundle(continuity_package)

        attempts: List[Dict[str, Any]] = []
        best_attempt: Optional[Dict[str, Any]] = None

        # ----------------------------------------------------------
        # initial attempt + retry attempts
        # ----------------------------------------------------------
        for retry_index in range(0, self.max_retries + 1):
            if retry_index == 0:
                current_bundle = copy.deepcopy(base_prompt_bundle)
            else:
                current_bundle = build_retry_prompt_bundle(
                    original_prompt_bundle=copy.deepcopy(
                        best_attempt["prompt_bundle"] if best_attempt else base_prompt_bundle
                    ),
                    scene_packet=scene_packet,
                    continuity_payload=self._build_retry_continuity_payload(
                        best_attempt,
                        continuity_package,
                    ),
                    retry_index=retry_index,
                )

            generation_result = self._generate_with_backend(
                backend=self.backend,
                prompt_bundle=current_bundle,
                scene_id=scene_id,
                repair_mode=False,
            )

            score_report = self._score_generation(
                scene_packet=scene_packet,
                generation_result=generation_result,
                prompt_bundle=current_bundle,
                continuity_package=continuity_package,
            )

            accepted = self._is_generation_accepted(score_report)
            drift_report = self._extract_drift_report(score_report)
            failure_tags = self._extract_failure_tags(score_report, drift_report)

            current_bundle = self._attach_retry_feedback(
                prompt_bundle=current_bundle,
                retry_index=retry_index,
                failure_tags=failure_tags,
                drift_report=drift_report,
            )

            attempt = {
                "retry_index": retry_index,
                "repair_mode": False,
                "accepted": accepted,
                "prompt_bundle": current_bundle,
                "generation_result": generation_result,
                "score_report": score_report,
                "drift_report": drift_report,
                "failure_tags": failure_tags,
                "backend_used": self._extract_backend_used(generation_result, self.backend),
                "score_value": self._score_value(score_report),
            }
            attempts.append(attempt)

            best_attempt = self._choose_better_attempt(best_attempt, attempt)

            if accepted:
                return self._finalize_attempt(
                    scene_id=scene_id,
                    selected_attempt=attempt,
                    attempts=attempts,
                    ok=True,
                )

        # ----------------------------------------------------------
        # repair stage (optional)
        # ----------------------------------------------------------
        if self.repair_after_retry:
            repair_attempt = self._run_repair_attempt(
                scene_id=scene_id,
                scene_packet=scene_packet,
                continuity_package=continuity_package,
                base_attempt=best_attempt if best_attempt is not None else attempts[-1],
            )
            if repair_attempt is not None:
                attempts.append(repair_attempt)
                best_attempt = self._choose_better_attempt(best_attempt, repair_attempt)

                if repair_attempt.get("accepted", False):
                    return self._finalize_attempt(
                        scene_id=scene_id,
                        selected_attempt=repair_attempt,
                        attempts=attempts,
                        ok=True,
                    )

        # ----------------------------------------------------------
        # fallback return with best available attempt
        # ----------------------------------------------------------
        if best_attempt is None and attempts:
            best_attempt = attempts[-1]

        if best_attempt is None:
            return {
                "ok": False,
                "accepted": False,
                "scene_id": scene_id,
                "generation_result": {},
                "score_report": {},
                "drift_report": {},
                "prompt_bundle": base_prompt_bundle,
                "retry_count": 0,
                "backend_used": "",
                "attempts": [],
                "repair_used": False,
                "error": "No generation attempt was produced.",
            }

        return self._finalize_attempt(
            scene_id=scene_id,
            selected_attempt=best_attempt,
            attempts=attempts,
            ok=False,
        )

    # ---------------------------------------------------------------------
    # prompt bundle resolution
    # ---------------------------------------------------------------------

    def _resolve_base_prompt_bundle(
        self,
        continuity_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve starting prompt bundle.

        Priority:
        1. continuity_package["prompt_bundle_override"]
        2. continuity_package["prompt_bundle"]
        3. build from continuity_package

        This is the key patch needed for SceneRepairEngine integration.
        """
        continuity_package = self._safe_dict(continuity_package)

        override_bundle = self._safe_dict(
            continuity_package.get("prompt_bundle_override", {})
        )
        if override_bundle:
            return self._merge_bundle_with_continuity_package(
                prompt_bundle=override_bundle,
                continuity_package=continuity_package,
            )

        existing_bundle = self._safe_dict(
            continuity_package.get("prompt_bundle", {})
        )
        if existing_bundle:
            return self._merge_bundle_with_continuity_package(
                prompt_bundle=existing_bundle,
                continuity_package=continuity_package,
            )

        built_bundle = build_prompt_bundle(continuity_package)
        return self._merge_bundle_with_continuity_package(
            prompt_bundle=built_bundle,
            continuity_package=continuity_package,
        )

    def _merge_bundle_with_continuity_package(
        self,
        prompt_bundle: Dict[str, Any],
        continuity_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make sure externally provided bundles still inherit current scene fields.
        """
        prompt_bundle = copy.deepcopy(self._safe_dict(prompt_bundle))
        continuity_package = self._safe_dict(continuity_package)

        # top-level identifiers
        if continuity_package.get("scene_id") and not prompt_bundle.get("scene_id"):
            prompt_bundle["scene_id"] = continuity_package["scene_id"]

        # always refresh these from continuity package if present
        for key in [
            "scene_packet",
            "same_constraints",
            "change_constraints",
            "generation_contract",
            "reference_bundle",
            "control_weights",
        ]:
            if key in continuity_package and continuity_package.get(key) not in [None, {}, []]:
                prompt_bundle[key] = copy.deepcopy(continuity_package[key])

        # generation params may come from repair policy
        if continuity_package.get("generation_params") not in [None, {}, []]:
            prompt_bundle["generation_params"] = copy.deepcopy(
                continuity_package["generation_params"]
            )

        # retry index / failure tags may come from repair engine
        prompt_metadata = self._safe_dict(prompt_bundle.get("prompt_metadata", {}))
        retry_context = self._safe_dict(prompt_bundle.get("retry_context", {}))

        if "retry_index" in continuity_package:
            prompt_metadata["retry_index"] = int(continuity_package.get("retry_index", 0) or 0)
            retry_context["retry_index"] = int(continuity_package.get("retry_index", 0) or 0)

        if "failure_tags" in continuity_package:
            prompt_metadata["failure_tags"] = self._safe_list(
                continuity_package.get("failure_tags", [])
            )
            retry_context["failure_tags"] = self._safe_list(
                continuity_package.get("failure_tags", [])
            )

        prompt_bundle["prompt_metadata"] = prompt_metadata
        prompt_bundle["retry_context"] = retry_context

        return prompt_bundle

    # ---------------------------------------------------------------------
    # core generation helpers
    # ---------------------------------------------------------------------

    def _run_repair_attempt(
        self,
        scene_id: str,
        scene_packet: Dict[str, Any],
        continuity_package: Dict[str, Any],
        base_attempt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        repair_backend = self.repair_backend if self.repair_backend is not None else self.backend
        if repair_backend is None:
            return None

        base_prompt_bundle = self._safe_dict(base_attempt.get("prompt_bundle", {}))
        drift_report = self._safe_dict(base_attempt.get("drift_report", {}))
        failure_tags = self._safe_list(base_attempt.get("failure_tags", []))

        repair_bundle = build_repair_prompt_bundle(
            continuity_package=copy.deepcopy(continuity_package),
            drift_report=drift_report,
            retry_index=int(base_attempt.get("retry_index", 0) or 0) + 1,
            failure_tags=failure_tags,
        )

        # keep generation params / refs / control weights from strongest retry attempt if available
        if base_prompt_bundle.get("generation_params"):
            repair_bundle["generation_params"] = copy.deepcopy(base_prompt_bundle["generation_params"])
        if base_prompt_bundle.get("control_weights"):
            repair_bundle["control_weights"] = copy.deepcopy(base_prompt_bundle["control_weights"])
        if base_prompt_bundle.get("reference_bundle"):
            repair_bundle["reference_bundle"] = copy.deepcopy(base_prompt_bundle["reference_bundle"])
        if base_prompt_bundle.get("retry_context"):
            repair_bundle["retry_context"] = copy.deepcopy(base_prompt_bundle["retry_context"])

        if self.strict_repair_mode:
            repair_bundle["positive_prompt"] = self._append_text(
                repair_bundle.get("positive_prompt", ""),
                "Repair mode: preserve already-correct content and only fix inconsistent identity, location, prop, style, or transition elements."
            )
            repair_bundle["repair_prompt"] = self._append_text(
                repair_bundle.get("repair_prompt", ""),
                "Preserve scene semantics while reducing continuity drift."
            )

        generation_result = self._generate_with_backend(
            backend=repair_backend,
            prompt_bundle=repair_bundle,
            scene_id=scene_id,
            repair_mode=True,
        )

        score_report = self._score_generation(
            scene_packet=scene_packet,
            generation_result=generation_result,
            prompt_bundle=repair_bundle,
            continuity_package=continuity_package,
        )
        accepted = self._is_generation_accepted(score_report)
        drift_report = self._extract_drift_report(score_report)
        failure_tags = self._extract_failure_tags(score_report, drift_report)

        repair_bundle = self._attach_retry_feedback(
            prompt_bundle=repair_bundle,
            retry_index=int(repair_bundle.get("prompt_metadata", {}).get("retry_index", 0) or 0),
            failure_tags=failure_tags,
            drift_report=drift_report,
        )

        return {
            "retry_index": int(repair_bundle.get("prompt_metadata", {}).get("retry_index", 0) or 0),
            "repair_mode": True,
            "accepted": accepted,
            "prompt_bundle": repair_bundle,
            "generation_result": generation_result,
            "score_report": score_report,
            "drift_report": drift_report,
            "failure_tags": failure_tags,
            "backend_used": self._extract_backend_used(generation_result, repair_backend),
            "score_value": self._score_value(score_report),
        }

    def _generate_with_backend(
        self,
        backend: Any,
        prompt_bundle: Dict[str, Any],
        scene_id: str,
        repair_mode: bool,
    ) -> Dict[str, Any]:
        if backend is None:
            return {
                "ok": False,
                "scene_id": scene_id,
                "error": "Backend is not available.",
            }

        # prefer explicit repair path if backend supports it
        if repair_mode and hasattr(backend, "generate_repair"):
            try:
                out = backend.generate_repair(prompt_bundle)
                return self._normalize_generation_result(out, scene_id, repair_mode)
            except Exception as e:
                return self._normalize_generation_result(
                    {
                        "ok": False,
                        "error": f"generate_repair failed: {e}",
                    },
                    scene_id,
                    repair_mode,
                )

        try:
            out = backend.generate(prompt_bundle)
            return self._normalize_generation_result(out, scene_id, repair_mode)
        except Exception as e:
            return self._normalize_generation_result(
                {
                    "ok": False,
                    "error": f"generate failed: {e}",
                },
                scene_id,
                repair_mode,
            )

    def _normalize_generation_result(
        self,
        result: Any,
        scene_id: str,
        repair_mode: bool,
    ) -> Dict[str, Any]:
        result = self._safe_dict(result)

        result.setdefault("scene_id", scene_id)
        result.setdefault("ok", True if result else False)
        result.setdefault("repair_mode", repair_mode)

        # normalize common video/keyframe fields
        result.setdefault("video_path", result.get("output_video_path", ""))
        result.setdefault(
            "keyframe_path",
            result.get("best_keyframe_path", result.get("selected_keyframe_path", ""))
        )
        result.setdefault("first_frame_path", result.get("first_frame_path", ""))
        result.setdefault("middle_frame_path", result.get("middle_frame_path", ""))
        result.setdefault("last_frame_path", result.get("last_frame_path", ""))

        metadata = self._safe_dict(result.get("metadata", {}))
        metadata.setdefault("generated_summary", self._safe_dict(metadata.get("generated_summary", {})))
        metadata.setdefault(
            "semantic_evidence_status",
            _safe_str(metadata.get("semantic_evidence_status", "missing")) or "missing"
        )
        metadata.setdefault(
            "has_semantic_evidence",
            self._metadata_has_semantic_evidence(metadata)
        )
        metadata.setdefault(
            "placeholder_conditioning_only",
            bool(metadata.get("placeholder_conditioning_only", False))
        )
        metadata.setdefault(
            "generation_failed",
            bool(metadata.get("generation_failed", False))
        )
        metadata.setdefault(
            "is_fallback_output",
            bool(metadata.get("is_fallback_output", False))
        )
        metadata.setdefault(
            "used_fallback_anchor",
            bool(metadata.get("used_fallback_anchor", False))
        )
        metadata.setdefault(
            "quality_score",
            self._safe_float(metadata.get("quality_score", 0.0), 0.0)
        )
        metadata.setdefault(
            "reference_source_type",
            _safe_str(metadata.get("reference_source_type", "missing")) or "missing"
        )
        metadata.setdefault("debug_error_stage", _safe_str(metadata.get("debug_error_stage", "")))
        metadata.setdefault("debug_error_text", _safe_str(metadata.get("debug_error_text", "")))
        result["metadata"] = metadata

        return result

    # ---------------------------------------------------------------------
    # scoring / acceptance
    # ---------------------------------------------------------------------

    def _score_generation(
        self,
        scene_packet: Dict[str, Any],
        generation_result: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        continuity_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.scorer is None:
            return self._default_score_report(generation_result)

        # try richer signatures first, then fall back
        try:
            score_report = self.scorer.score(
                scene_packet=scene_packet,
                generation_result=generation_result,
                prompt_bundle=prompt_bundle,
                continuity_package=continuity_package,
            )
            return self._normalize_score_report(score_report, generation_result)
        except TypeError:
            pass
        except Exception as e:
            return self._normalize_score_report(
                {
                    "ok": False,
                    "error": f"scorer error: {e}",
                },
                generation_result,
            )

        try:
            score_report = self.scorer.score(
                scene_packet,
                generation_result,
                prompt_bundle,
                continuity_package,
            )
            return self._normalize_score_report(score_report, generation_result)
        except TypeError:
            pass
        except Exception as e:
            return self._normalize_score_report(
                {
                    "ok": False,
                    "error": f"scorer error: {e}",
                },
                generation_result,
            )

        try:
            score_report = self.scorer.score(scene_packet, generation_result)
            return self._normalize_score_report(score_report, generation_result)
        except Exception as e:
            return self._normalize_score_report(
                {
                    "ok": False,
                    "error": f"scorer error: {e}",
                },
                generation_result,
            )

    def _normalize_score_report(
        self,
        score_report: Any,
        generation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        score_report = self._safe_dict(score_report)
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        score_report.setdefault("ok", True)
        score_report.setdefault(
            "overall_score",
            self._safe_float(score_report.get("score", 0.0), 0.0)
        )
        score_report.setdefault("score", score_report.get("overall_score", 0.0))
        score_report.setdefault(
            "accepted",
            self._safe_bool(score_report.get("accepted", None), None)
        )
        score_report.setdefault("hard_fail", False)

        drift_report = self._safe_dict(score_report.get("drift_report", {}))
        drift_report.setdefault(
            "drift_flags",
            self._safe_list(drift_report.get("drift_flags", []))
        )
        drift_report.setdefault("severity", drift_report.get("severity", "low"))

        semantic_evidence_status = _safe_str(
            score_report.get(
                "semantic_evidence_status",
                drift_report.get(
                    "semantic_evidence_status",
                    metadata.get("semantic_evidence_status", "missing"),
                ),
            )
        ).lower() or "missing"

        has_semantic_evidence = bool(
            score_report.get(
                "has_semantic_evidence",
                drift_report.get(
                    "has_semantic_evidence",
                    metadata.get("has_semantic_evidence", self._metadata_has_semantic_evidence(metadata)),
                ),
            )
        )

        placeholder_conditioning_only = bool(
            score_report.get(
                "placeholder_conditioning_only",
                drift_report.get(
                    "placeholder_conditioning_only",
                    metadata.get("placeholder_conditioning_only", False),
                ),
            )
        )

        score_report["semantic_evidence_status"] = semantic_evidence_status
        score_report["has_semantic_evidence"] = has_semantic_evidence
        score_report["placeholder_conditioning_only"] = placeholder_conditioning_only

        drift_report["semantic_evidence_status"] = semantic_evidence_status
        drift_report["has_semantic_evidence"] = has_semantic_evidence
        drift_report["placeholder_conditioning_only"] = placeholder_conditioning_only

        score_report["drift_report"] = drift_report
        return score_report

    def _default_score_report(self, generation_result: Dict[str, Any]) -> Dict[str, Any]:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        ok = bool(generation_result.get("ok", False))
        semantic_evidence_status = _safe_str(
            metadata.get("semantic_evidence_status", "missing")
        ).lower() or "missing"
        has_semantic_evidence = bool(
            metadata.get("has_semantic_evidence", self._metadata_has_semantic_evidence(metadata))
        )
        placeholder_conditioning_only = bool(
            metadata.get("placeholder_conditioning_only", False)
        )

        if not ok:
            base_score = 0.0
            accepted = False
            hard_fail = True
            drift_flags = ["generation_failure"]
            severity = "high"
        elif placeholder_conditioning_only:
            base_score = 0.05
            accepted = False
            hard_fail = True
            drift_flags = ["placeholder_conditioning", "missing_semantic_evidence"]
            severity = "high"
        elif semantic_evidence_status == "missing" or not has_semantic_evidence:
            base_score = 0.20
            accepted = False
            hard_fail = False
            drift_flags = ["missing_semantic_evidence"]
            severity = "medium"
        else:
            base_score = 0.70
            accepted = base_score >= self.accept_score_threshold
            hard_fail = False
            drift_flags = []
            severity = "low"

        return {
            "ok": ok,
            "overall_score": base_score,
            "score": base_score,
            "accepted": accepted,
            "hard_fail": hard_fail,
            "semantic_evidence_status": semantic_evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "placeholder_conditioning_only": placeholder_conditioning_only,
            "drift_report": {
                "drift_flags": drift_flags,
                "severity": severity,
                "semantic_evidence_status": semantic_evidence_status,
                "has_semantic_evidence": has_semantic_evidence,
                "placeholder_conditioning_only": placeholder_conditioning_only,
            },
        }

    def _is_generation_accepted(self, score_report: Dict[str, Any]) -> bool:
        score_report = self._safe_dict(score_report)

        accepted = score_report.get("accepted", None)
        hard_fail = bool(score_report.get("hard_fail", False))
        if hard_fail:
            return False

        drift_report = self._safe_dict(score_report.get("drift_report", {}))
        drift_flags = [
            str(x).lower()
            for x in self._safe_list(drift_report.get("drift_flags", []))
        ]
        evidence_status = _safe_str(
            score_report.get(
                "semantic_evidence_status",
                drift_report.get("semantic_evidence_status", "")
            )
        ).lower()
        has_semantic_evidence = bool(
            score_report.get(
                "has_semantic_evidence",
                drift_report.get("has_semantic_evidence", False)
            )
        )
        placeholder_conditioning_only = bool(
            score_report.get(
                "placeholder_conditioning_only",
                drift_report.get("placeholder_conditioning_only", False)
            )
        )

        if placeholder_conditioning_only:
            return False
        if evidence_status == "missing" and not has_semantic_evidence:
            return False
        if "placeholder_conditioning" in drift_flags:
            return False
        if "missing_semantic_evidence" in drift_flags:
            return False

        if self.hard_fail_on_identity and "identity_drift" in drift_flags:
            return False
        if self.hard_fail_on_location and "location_drift" in drift_flags:
            return False

        overall_score = self._score_value(score_report)

        if isinstance(accepted, bool):
            if accepted and overall_score >= self.accept_score_threshold:
                return True
            if accepted is False:
                return False

        return overall_score >= self.accept_score_threshold

    # ---------------------------------------------------------------------
    # attempt comparison / finalization
    # ---------------------------------------------------------------------

    def _choose_better_attempt(
        self,
        best_attempt: Optional[Dict[str, Any]],
        new_attempt: Dict[str, Any],
    ) -> Dict[str, Any]:
        if best_attempt is None:
            return new_attempt

        if new_attempt.get("accepted", False) and not best_attempt.get("accepted", False):
            return new_attempt
        if best_attempt.get("accepted", False) and not new_attempt.get("accepted", False):
            return best_attempt

        best_score = self._safe_float(best_attempt.get("score_value", 0.0), 0.0)
        new_score = self._safe_float(new_attempt.get("score_value", 0.0), 0.0)
        if new_score > best_score:
            return new_attempt

        best_severity = self._drift_rank(best_attempt.get("drift_report", {}))
        new_severity = self._drift_rank(new_attempt.get("drift_report", {}))
        if new_score == best_score and new_severity < best_severity:
            return new_attempt

        return best_attempt

    def _finalize_attempt(
        self,
        scene_id: str,
        selected_attempt: Dict[str, Any],
        attempts: List[Dict[str, Any]],
        ok: bool,
    ) -> Dict[str, Any]:
        generation_result = self._safe_dict(selected_attempt.get("generation_result", {}))
        score_report = self._safe_dict(selected_attempt.get("score_report", {}))
        drift_report = self._safe_dict(selected_attempt.get("drift_report", {}))
        prompt_bundle = self._safe_dict(selected_attempt.get("prompt_bundle", {}))

        metadata = self._safe_dict(generation_result.get("metadata", {}))
        metadata["selected_retry_index"] = int(selected_attempt.get("retry_index", 0) or 0)
        metadata["selected_repair_mode"] = bool(selected_attempt.get("repair_mode", False))
        metadata["backend_used"] = _safe_str(selected_attempt.get("backend_used", ""))
        metadata["failure_tags"] = self._safe_list(selected_attempt.get("failure_tags", []))
        metadata["overall_score"] = self._score_value(score_report)
        metadata["semantic_evidence_status"] = _safe_str(
            score_report.get(
                "semantic_evidence_status",
                drift_report.get(
                    "semantic_evidence_status",
                    metadata.get("semantic_evidence_status", "missing")
                )
            )
        ) or "missing"
        metadata["has_semantic_evidence"] = bool(
            score_report.get(
                "has_semantic_evidence",
                drift_report.get("has_semantic_evidence", False)
            )
        )
        metadata["placeholder_conditioning_only"] = bool(
            metadata.get("placeholder_conditioning_only", False)
            or score_report.get("placeholder_conditioning_only", False)
            or drift_report.get("placeholder_conditioning_only", False)
        )
        metadata["acceptance_blocked_by_evidence"] = bool(
            metadata.get("placeholder_conditioning_only", False)
            or metadata["semantic_evidence_status"] == "missing"
        )
        generation_result["metadata"] = metadata

        return {
            "ok": ok,
            "accepted": bool(selected_attempt.get("accepted", False)),
            "scene_id": scene_id,
            "generation_result": generation_result,
            "score_report": score_report,
            "drift_report": drift_report,
            "prompt_bundle": prompt_bundle,
            "retry_count": int(selected_attempt.get("retry_index", 0) or 0),
            "backend_used": _safe_str(selected_attempt.get("backend_used", "")),
            "attempts": [self._attempt_summary(x) for x in attempts],
            "repair_used": bool(selected_attempt.get("repair_mode", False)),
        }

    def _attempt_summary(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        generation_result = self._safe_dict(attempt.get("generation_result", {}))
        metadata = self._safe_dict(generation_result.get("metadata", {}))
        score_report = self._safe_dict(attempt.get("score_report", {}))
        drift_report = self._safe_dict(attempt.get("drift_report", {}))
        return {
            "retry_index": int(attempt.get("retry_index", 0) or 0),
            "repair_mode": bool(attempt.get("repair_mode", False)),
            "accepted": bool(attempt.get("accepted", False)),
            "score": self._safe_float(attempt.get("score_value", 0.0), 0.0),
            "backend_used": _safe_str(attempt.get("backend_used", "")),
            "failure_tags": self._safe_list(attempt.get("failure_tags", [])),
            "video_path": _safe_str(generation_result.get("video_path", "")),
            "keyframe_path": _safe_str(generation_result.get("keyframe_path", "")),
            "generation_failed": bool(metadata.get("generation_failed", False)),
            "is_fallback_output": bool(metadata.get("is_fallback_output", False)),
            "used_fallback_anchor": bool(metadata.get("used_fallback_anchor", False)),
            "placeholder_conditioning_only": bool(metadata.get("placeholder_conditioning_only", False)),
            "quality_score": self._safe_float(metadata.get("quality_score", 0.0), 0.0),
            "used_init_reference": _safe_str(metadata.get("used_init_reference", "")),
            "reference_source_type": _safe_str(metadata.get("reference_source_type", "missing")) or "missing",
            "semantic_evidence_status": _safe_str(
                score_report.get(
                    "semantic_evidence_status",
                    drift_report.get(
                        "semantic_evidence_status",
                        metadata.get("semantic_evidence_status", "missing")
                    )
                )
            ) or "missing",
            "has_semantic_evidence": bool(
                score_report.get(
                    "has_semantic_evidence",
                    drift_report.get("has_semantic_evidence", False)
                )
            ),
            "debug_error_stage": _safe_str(metadata.get("debug_error_stage", "")),
            "debug_error_text": _safe_str(metadata.get("debug_error_text", "")),
        }

    # ---------------------------------------------------------------------
    # retry context / feedback helpers
    # ---------------------------------------------------------------------

    def _build_retry_continuity_payload(
        self,
        best_attempt: Optional[Dict[str, Any]],
        continuity_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "generation_contract": self._safe_dict(continuity_package.get("generation_contract", {})),
            "reference_bundle": self._safe_dict(continuity_package.get("reference_bundle", {})),
            "control_weights": self._safe_dict(continuity_package.get("control_weights", {})),
            "failure_tags": [],
            "drift_report": {},
        }

        if best_attempt is None:
            return payload

        payload["failure_tags"] = self._safe_list(best_attempt.get("failure_tags", []))
        payload["drift_report"] = self._safe_dict(best_attempt.get("drift_report", {}))

        prev_bundle = self._safe_dict(best_attempt.get("prompt_bundle", {}))
        if prev_bundle.get("reference_bundle"):
            payload["reference_bundle"] = copy.deepcopy(prev_bundle["reference_bundle"])
        if prev_bundle.get("control_weights"):
            payload["control_weights"] = copy.deepcopy(prev_bundle["control_weights"])

        return payload

    def _attach_retry_feedback(
        self,
        prompt_bundle: Dict[str, Any],
        retry_index: int,
        failure_tags: List[str],
        drift_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_bundle = copy.deepcopy(self._safe_dict(prompt_bundle))
        metadata = self._safe_dict(prompt_bundle.get("prompt_metadata", {}))
        metadata["retry_index"] = retry_index
        metadata["failure_tags"] = self._safe_list(failure_tags)
        metadata["drift_severity"] = self._safe_dict(drift_report).get("severity", "low")
        prompt_bundle["prompt_metadata"] = metadata

        retry_context = self._safe_dict(prompt_bundle.get("retry_context", {}))
        retry_context["retry_index"] = retry_index
        retry_context["failure_tags"] = self._safe_list(failure_tags)
        retry_context["drift_report"] = self._safe_dict(drift_report)
        prompt_bundle["retry_context"] = retry_context
        return prompt_bundle

    # ---------------------------------------------------------------------
    # extractors
    # ---------------------------------------------------------------------

    def _extract_drift_report(self, score_report: Dict[str, Any]) -> Dict[str, Any]:
        drift_report = self._safe_dict(score_report.get("drift_report", {}))
        drift_report.setdefault("drift_flags", self._safe_list(drift_report.get("drift_flags", [])))
        drift_report.setdefault("severity", drift_report.get("severity", "low"))
        drift_report.setdefault(
            "semantic_evidence_status",
            _safe_str(drift_report.get("semantic_evidence_status", "missing")) or "missing"
        )
        drift_report.setdefault(
            "has_semantic_evidence",
            bool(drift_report.get("has_semantic_evidence", False))
        )
        drift_report.setdefault(
            "placeholder_conditioning_only",
            bool(drift_report.get("placeholder_conditioning_only", False))
        )
        return drift_report

    def _extract_failure_tags(
        self,
        score_report: Dict[str, Any],
        drift_report: Dict[str, Any],
    ) -> List[str]:
        tags = []
        tags.extend(self._safe_list(score_report.get("failure_tags", [])))
        tags.extend(self._safe_list(self._safe_dict(drift_report).get("drift_flags", [])))

        if not bool(score_report.get("ok", True)):
            tags.append("generation_failure")
        if self._score_value(score_report) < self.accept_score_threshold:
            tags.append("low_score")

        if bool(score_report.get("placeholder_conditioning_only", False)):
            tags.append("placeholder_conditioning")
        if _safe_str(score_report.get("semantic_evidence_status", "missing")).lower() == "missing":
            tags.append("missing_semantic_evidence")

        return self._dedupe_str_list(tags)

    def _extract_backend_used(self, generation_result: Dict[str, Any], backend: Any) -> str:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        for key in ["backend_used", "backend", "route_backend"]:
            val = _safe_str(metadata.get(key, "")) or _safe_str(generation_result.get(key, ""))
            if val:
                return val

        name = getattr(backend, "__class__", type("X", (), {})).__name__
        return _safe_str(name)

    # ---------------------------------------------------------------------
    # small utilities
    # ---------------------------------------------------------------------

    def _scene_id(self, continuity_package: Dict[str, Any]) -> str:
        return _safe_str(continuity_package.get("scene_id", ""))

    def _score_value(self, score_report: Dict[str, Any]) -> float:
        score_report = self._safe_dict(score_report)
        if "overall_score" in score_report:
            return self._safe_float(score_report.get("overall_score", 0.0), 0.0)
        if "score" in score_report:
            return self._safe_float(score_report.get("score", 0.0), 0.0)
        return 0.0

    def _drift_rank(self, drift_report: Dict[str, Any]) -> int:
        severity = _safe_str(self._safe_dict(drift_report).get("severity", "low")).lower()
        if severity == "none":
            return 0
        if severity == "low":
            return 1
        if severity == "medium":
            return 2
        if severity == "high":
            return 3
        return 2

    def _append_text(self, base: str, extra: str) -> str:
        base = _safe_str(base)
        extra = _safe_str(extra)
        if not base:
            return extra
        if not extra:
            return base
        return (base + " " + extra).strip()

    def _dedupe_str_list(self, items: List[Any]) -> List[str]:
        seen = set()
        out = []
        for item in items or []:
            text = _safe_str(item)
            if not text:
                continue
            low = text.lower()
            if low not in seen:
                seen.add(low)
                out.append(text)
        return out

    def _metadata_has_semantic_evidence(self, metadata: Dict[str, Any]) -> bool:
        metadata = self._safe_dict(metadata)
        generated_summary = self._safe_dict(metadata.get("generated_summary", {}))
        if not generated_summary:
            return False

        characters = self._safe_list(generated_summary.get("characters", []))
        location = self._safe_dict(generated_summary.get("location", {}))
        props = self._safe_list(generated_summary.get("props", []))
        style = self._safe_dict(generated_summary.get("style", {}))

        return bool(characters or location or props or style)

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {}

    def _safe_list(self, x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return []

    def _safe_bool(self, x: Any, default: Optional[bool] = False) -> Optional[bool]:
        if isinstance(x, bool):
            return x
        if x is None:
            return default
        return default

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()