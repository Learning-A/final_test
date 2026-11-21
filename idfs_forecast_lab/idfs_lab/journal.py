import math
import time

from .io import append_event


class PredictionJournal:
    def __init__(self, path, clock=time.time_ns):
        self.path, self.clock = path, clock

    def issue(self, predict, features, target, window_start_ns, window_end_ns,
              feature_end_ns, features_available_ns, model_sha256, protocol_hash):
        before = self.clock()
        output = predict(features)
        details=output if isinstance(output,dict) else {}
        value = float(details["prediction"] if details else output)
        if not math.isfinite(value) or value < 0:
            raise ValueError("prediction must be finite and nonnegative")
        # Capture AFTER computation. A slow model cannot pass by backdating issue time.
        issued = self.clock()
        event = {"event":"prediction","target":target,"window_start_ns":window_start_ns,
                 "window_end_ns":window_end_ns,"feature_end_ns":feature_end_ns,
                 "features_available_ns":features_available_ns,
                 "compute_start_ns":before,"issue_epoch_ns":issued,
                 "prediction_iops":value,"model_sha256":model_sha256,
                 "protocol_hash":protocol_hash,
                 "late":issued >= window_start_ns,
                 "causal":feature_end_ns <= issued and features_available_ns <= before,
                 "feature_vector":[float(x) for x in features] if isinstance(features,(list,tuple)) else features}
        if details:
            event.update(model_raw_iops=float(details["raw"]),calibration=details.get("calibration"),
                         mode=details.get("mode","online"),diagnostics=details.get("diagnostics"))
        append_event(self.path,event)
        committed=self.clock()
        append_event(self.path,{"event":"prediction_commit","target":target,
                               "window_start_ns":window_start_ns,"committed_epoch_ns":committed})
        return dict(event,persisted_epoch_ns=committed,late=issued>=window_start_ns or committed>=window_start_ns)
