"""Phase 5: Musculoskeletal Model Scaling & Inverse Kinematics Runner.

Fits a subject-scaled OpenSim musculoskeletal model (e.g. Rajagopal_Pose2Sim)
to the triangulated and filtered .trc marker trajectories to calculate physiological joint angles.
Supports direct OpenSim InverseKinematicsTool execution and full Pose2Sim pipeline scaling + IK.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

from Pose2Sim import Pose2Sim
import opensim as osim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pose2sim_kinematics(config_path: Union[str, Path] = "Config.toml") -> None:
    """Runs subject scaling and Inverse Kinematics through Pose2Sim."""
    config_path = str(config_path)
    logger.info(f"Running Pose2Sim.kinematics('{config_path}')...")
    Pose2Sim.kinematics(config_path)
    logger.info("Pose2Sim kinematics execution complete.")


def execute_opensim_ik(
    model_path: Union[str, Path],
    trc_path: Union[str, Path],
    output_mot_path: Union[str, Path],
    ik_setup_xml: Optional[Union[str, Path]] = None,
    accuracy: float = 1e-4,
    constraint_weight: float = 20.0,
) -> None:
    """Executes OpenSim Inverse Kinematics tool on an experimental TRC marker file.

    Args:
        model_path: Path to scaled OpenSim model (.osim).
        trc_path: Path to filtered 3D marker coordinates (.trc).
        output_mot_path: Path for output motion file (.mot).
        ik_setup_xml: Optional path to pre-configured IK_Setup.xml.
        accuracy: IK numerical solver convergence accuracy.
        constraint_weight: Weight for model kinematic constraints.
    """
    model_path = Path(model_path).resolve()
    trc_path = Path(trc_path).resolve()
    output_mot_path = Path(output_mot_path).resolve()
    output_mot_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"OpenSim model file not found: {model_path}")
    if not trc_path.is_file():
        raise FileNotFoundError(f"TRC marker file not found: {trc_path}")

    logger.info(f"Loading OpenSim model from: {model_path}")
    model = osim.Model(str(model_path))

    # If the model has 0 markers (e.g. unscaled base model), attach markers
    if model.getMarkerSet().getSize() == 0:
        markers_xml = Path("configs/opensim_settings/Markers_Halpe26.xml").resolve()
        if markers_xml.is_file():
            logger.info(f"Attaching markers from: {markers_xml}")
            marker_set = osim.MarkerSet(str(markers_xml))
            for i in range(marker_set.getSize()):
                model.addMarker(marker_set.get(i).clone())

    model.initSystem()

    if ik_setup_xml and Path(ik_setup_xml).is_file():
        ik_setup_xml = Path(ik_setup_xml).resolve()
        logger.info(f"Loading IK Tool setup from XML: {ik_setup_xml}")
        ik_tool = osim.InverseKinematicsTool(str(ik_setup_xml))
    else:
        logger.info("Initializing programmatic InverseKinematicsTool...")
        ik_tool = osim.InverseKinematicsTool()

    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(str(trc_path))
    ik_tool.setOutputMotionFileName(str(output_mot_path))

    # Support both snake_case (OpenSim 4.4+) and camelCase method bindings
    if hasattr(ik_tool, "set_accuracy"):
        ik_tool.set_accuracy(accuracy)
    elif hasattr(ik_tool, "setAccuracy"):
        ik_tool.setAccuracy(accuracy)

    if hasattr(ik_tool, "set_constraint_weight"):
        ik_tool.set_constraint_weight(constraint_weight)
    elif hasattr(ik_tool, "setConstraintWeight"):
        ik_tool.setConstraintWeight(constraint_weight)

    # Automatically derive start and end time from TRC marker data
    marker_data = osim.MarkerData(str(trc_path))
    start_time = marker_data.getStartFrameTime()
    end_time = marker_data.getLastFrameTime()
    ik_tool.setStartTime(start_time)
    ik_tool.setEndTime(end_time)

    logger.info(
        f"Running OpenSim Inverse Kinematics over interval [{start_time:.3f}s, {end_time:.3f}s]..."
    )
    ik_tool.run()
    logger.info(f"Kinematics successfully computed and saved to: {output_mot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: OpenSim Inverse Kinematics Runner")
    parser.add_argument("--config", type=str, default="Config.toml", help="Path to Pose2Sim Config.toml")
    parser.add_argument("--pipeline", action="store_true", help="Run full Pose2Sim scaling + IK pipeline")
    parser.add_argument("--model", type=str, default="models/Rajagopal_Pose2Sim.osim", help="Path to .osim model")
    parser.add_argument("--trc", type=str, default="", help="Path to .trc marker file (auto-detects latest if empty)")
    parser.add_argument("--output-mot", type=str, default="opensim/Results/trial_kinematics.mot", help="Output .mot motion file")
    parser.add_argument("--ik-setup", type=str, default="configs/opensim_settings/IK_Setup_Halpe26.xml", help="Optional IK setup XML")

    args = parser.parse_args()

    if args.pipeline:
        run_pose2sim_kinematics(config_path=args.config)
        return

    # Auto-detect TRC if not specified
    trc_path = args.trc
    if not trc_path:
        trc_candidates = list(Path("pose-3d").glob("*_LSTM.trc")) or list(Path("pose-3d").glob("*.trc"))
        if trc_candidates:
            trc_path = str(max(trc_candidates, key=lambda p: p.stat().st_ctime))
            logger.info(f"Auto-selected TRC marker file: {trc_path}")
        else:
            trc_path = "pose-3d/video2motion_0-899_filt_butterworth_LSTM.trc"

    # If scaled model exists in kinematics/, use it by default
    model_path = args.model
    scaled_models = list(Path("kinematics").glob("*_scaled.osim"))
    if scaled_models and model_path == "models/Rajagopal_Pose2Sim.osim":
        model_path = str(max(scaled_models, key=lambda p: p.stat().st_ctime))
        logger.info(f"Found scaled model in kinematics/, using: {model_path}")

    execute_opensim_ik(
        model_path=model_path,
        trc_path=trc_path,
        output_mot_path=args.output_mot,
        ik_setup_xml=args.ik_setup if Path(args.ik_setup).is_file() else None,
    )


if __name__ == "__main__":
    main()
