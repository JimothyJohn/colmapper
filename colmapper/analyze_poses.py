import argparse
import numpy as np
import os

# AI-generated comment: Thresholds for considering a parameter significantly different in the summary.
# For H, W, F: a ratio outside (1 +/- SENSITIVITY_HWF)
# For Near, Far: a ratio outside (1 +/- SENSITIVITY_BOUNDS)
SENSITIVITY_HWF = 0.02  # 2%
SENSITIVITY_BOUNDS = 0.10  # 10%


def extract_camera_data(file_path):
    """
    Loads a .npy file, extracts key camera parameters, and calculates aggregate statistics.
    AI-generated comment: This function loads the .npy file, validates its structure,
    extracts H, W, F, Near, and Far for each camera, and then computes aggregate statistics
    (average, median, std, min, max) for these parameters across all cameras in the file.
    Returns a dictionary with raw data, per-camera parameters, and aggregate stats.
    """
    try:
        data = np.load(file_path)

        if data.ndim != 2 or data.shape[1] != 17:
            print(
                f"\n**Error (in `extract_camera_data` for `{file_path}`):** Data shape `{data.shape}` is not (N, 17)."
            )
            return None

        num_cameras = data.shape[0]
        cameras_params = []
        aggregates = {}

        if num_cameras == 0:
            # AI-generated comment: Populate aggregates with NaNs or zeros if no cameras, to prevent errors.
            for param in ["H", "W", "F", "Near", "Far"]:
                aggregates[param] = {
                    "avg": np.nan,
                    "med": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            return {
                "file_path": file_path,
                "num_cameras": 0,
                "cameras_params": [],
                "raw_data": data,
                "aggregates": aggregates,
            }

        # AI-generated comment: Extract per-camera parameters.
        param_values = {key: [] for key in ["H", "W", "F", "Near", "Far"]}
        for i in range(num_cameras):
            h, w, f, near, far = (
                data[i, 4],
                data[i, 9],
                data[i, 14],
                data[i, 15],
                data[i, 16],
            )
            cameras_params.append(
                {"H": h, "W": w, "F": f, "Near": near, "Far": far, "id": i + 1}
            )
            param_values["H"].append(h)
            param_values["W"].append(w)
            param_values["F"].append(f)
            param_values["Near"].append(near)
            param_values["Far"].append(far)

        # AI-generated comment: Calculate aggregate statistics for each parameter.
        for param_key, values_list in param_values.items():
            values_np = np.array(values_list)
            aggregates[param_key] = {
                "avg": np.mean(values_np),
                "med": np.median(values_np),
                "std": np.std(values_np),
                "min": np.min(values_np),
                "max": np.max(values_np),
            }

        return {
            "file_path": file_path,
            "num_cameras": num_cameras,
            "cameras_params": cameras_params,
            "raw_data": data,
            "aggregates": aggregates,
        }

    except FileNotFoundError:
        print(f"\n**Error:** File not found at `{file_path}`")
        return None
    except Exception as e:
        print(
            f"\n**An error occurred while processing `{file_path}` in `extract_camera_data`:** `{e}`"
        )
        return None


def generate_executive_summary(baseline_info, comparison_info):
    """
    Generates a Markdown executive summary comparing aggregate file statistics.
    AI-generated comment: This function constructs a summary that highlights key differences
    between a comparison file and the baseline by comparing their aggregate statistics (average, median)
    for H, W, F, Near, and Far parameters, based on predefined sensitivities.
    """
    summary_lines = []
    summary_lines.append(
        f"\n## Executive Summary: `{os.path.basename(comparison_info['file_path'])}` vs. Baseline (`{os.path.basename(baseline_info['file_path'])}`)"
    )

    if baseline_info["num_cameras"] != comparison_info["num_cameras"]:
        summary_lines.append(
            f"- **Camera Count Mismatch:** Baseline has `{baseline_info['num_cameras']}` cameras, "
            f"`{os.path.basename(comparison_info['file_path'])}` has `{comparison_info['num_cameras']}` cameras."
        )
    else:
        summary_lines.append(
            f"- **Camera Count:** Both files have `{baseline_info['num_cameras']}` cameras."
        )

    if baseline_info["num_cameras"] == 0 or comparison_info["num_cameras"] == 0:
        summary_lines.append(
            "- Cannot perform detailed parameter comparison as one or both files have no cameras."
        )
        return summary_lines

    param_definitions = {
        "H": {"name": "Height (H)", "sensitivity": SENSITIVITY_HWF},
        "W": {"name": "Width (W)", "sensitivity": SENSITIVITY_HWF},
        "F": {"name": "Focal Length (F)", "sensitivity": SENSITIVITY_HWF},
        "Near": {"name": "Near Bound", "sensitivity": SENSITIVITY_BOUNDS},
        "Far": {"name": "Far Bound", "sensitivity": SENSITIVITY_BOUNDS},
    }

    overall_differences_found = False
    for param_key, P_info in param_definitions.items():
        base_avg = baseline_info["aggregates"][param_key]["avg"]
        comp_avg = comparison_info["aggregates"][param_key]["avg"]
        base_med = baseline_info["aggregates"][param_key]["med"]
        comp_med = comparison_info["aggregates"][param_key]["med"]

        diff_detected_param = False
        # AI-generated comment: Compare average values.
        if (
            base_avg != 0 and not np.isnan(base_avg) and not np.isnan(comp_avg)
        ):  # Ensure base_avg is not zero and values are not NaN
            avg_ratio = comp_avg / base_avg
            if not (1 - P_info["sensitivity"] < avg_ratio < 1 + P_info["sensitivity"]):
                summary_lines.append(
                    f"- **{P_info['name']} (Average):** Differs significantly. "
                    f"Baseline Avg: `{base_avg:.2f}`, Comparison Avg: `{comp_avg:.2f}` (Ratio: `{avg_ratio:.2f}`)\n"
                )
                overall_differences_found = True
                diff_detected_param = True
        elif np.isnan(base_avg) or np.isnan(comp_avg):
            summary_lines.append(
                f"- **{P_info['name']} (Average):** Cannot compare due to NaN values in aggregates.\n"
            )
            overall_differences_found = True  # Count as a difference
            diff_detected_param = True

        # AI-generated comment: Optionally, compare median values if average didn't show difference or as additional info.
        if (
            not diff_detected_param
            and base_med != 0
            and not np.isnan(base_med)
            and not np.isnan(comp_med)
        ):
            med_ratio = comp_med / base_med
            if not (1 - P_info["sensitivity"] < med_ratio < 1 + P_info["sensitivity"]):
                summary_lines.append(
                    f"- **{P_info['name']} (Median):** Differs significantly. "
                    f"Baseline Median: `{base_med:.2f}`, Comparison Median: `{comp_med:.2f}` (Ratio: `{med_ratio:.2f}`)\n"
                )
                overall_differences_found = True
        elif not diff_detected_param and (np.isnan(base_med) or np.isnan(comp_med)):
            summary_lines.append(
                f"- **{P_info['name']} (Median):** Cannot compare due to NaN values in aggregates.\n"
            )
            overall_differences_found = True

    if (
        not overall_differences_found
        and baseline_info["num_cameras"] == comparison_info["num_cameras"]
    ):
        summary_lines.append(
            "- **Overall Aggregate Comparison:** No significant deviations detected in average/median H, W, F, Near, or Far parameters based on current sensitivities."
        )
    elif (
        not overall_differences_found
        and baseline_info["num_cameras"] != comparison_info["num_cameras"]
    ):
        summary_lines.append(
            "- **Overall Aggregate Comparison:** No significant deviations detected in average/median parameters for comparable cameras based on current sensitivities, beyond camera count mismatch."
        )

    summary_lines.append(
        f"    *(Note: Significance thresholds for Avg/Median comparison - H,W,F: +/-{SENSITIVITY_HWF*100:.0f}%, Near/Far: +/-{SENSITIVITY_BOUNDS*100:.0f}%)*"
    )
    return summary_lines


def generate_detailed_markdown_report(
    file_info, is_baseline=True, baseline_info_for_ratios=None
):
    """
    Generates a detailed Markdown report for a single .npy file, including aggregate stats.
    AI-generated comment: This function takes the structured data for one file and prints
    a comprehensive Markdown report. This includes general file info, aggregate statistics for the file,
    per-camera parameters, and relevant ratios (intra-file for baseline, or cross-file for comparison files).
    """
    file_path = file_info["file_path"]
    num_cameras = file_info["num_cameras"]
    cameras_params = file_info["cameras_params"]
    data = file_info["raw_data"]
    aggregates = file_info["aggregates"]

    print(f"\n## Detailed Report for: `{file_path}`")
    if not is_baseline and baseline_info_for_ratios is not None:
        print(
            f"### (Cross-File Ratios for individual cameras vs. Baseline: `{baseline_info_for_ratios['file_path']}`)"
        )

    print(f"- **Shape of the array:** `{data.shape}`")
    print(f"- **Data type of the array:** `{data.dtype}`")
    print(f"- **Number of cameras found:** `{num_cameras}`")

    if num_cameras > 0:
        print(f"### Aggregate Statistics for `{os.path.basename(file_path)}`")
        for param_key, stats in aggregates.items():
            p_name = param_key  # Simple name for now, could map to full name
            if param_key == "F":
                p_name = "Focal Length"
            elif param_key == "H":
                p_name = "Height"
            elif param_key == "W":
                p_name = "Width"

            print(f"  - **{p_name}:**")
            print(f"    - Average: `{stats['avg']:.2f}`")
            print(f"    - Median:  `{stats['med']:.2f}`")
            print(f"    - Std Dev: `{stats['std']:.2f}`")
            print(f"    - Min:     `{stats['min']:.2f}`")
            print(f"    - Max:     `{stats['max']:.2f}`")
    else:
        print(
            "\n**Warning:** No camera data to detail or provide aggregates for in this file."
        )
        return  # AI-generated comment: No further detailed per-camera reporting if no cameras.

    # AI-generated comment: Per-camera details section.
    print(f"\n### Per-Camera Details for `{os.path.basename(file_path)}`")
    file_base_h, file_base_w, file_base_f, file_base_near, file_base_far = (None,) * 5
    if num_cameras > 0 and is_baseline:  # For intra-file ratios if this is baseline
        first_cam = cameras_params[0]
        file_base_h, file_base_w, file_base_f, file_base_near, file_base_far = (
            first_cam["H"],
            first_cam["W"],
            first_cam["F"],
            first_cam["Near"],
            first_cam["Far"],
        )

    for i in range(num_cameras):
        cam_params = cameras_params[i]
        print(f"\n#### Camera {cam_params['id']} (in `{os.path.basename(file_path)}`)")
        print(f"  - **Height (H):**         `{cam_params['H']}`")
        print(f"  - **Width (W):**          `{cam_params['W']}`")
        print(f"  - **Focal Length (F):**   `{cam_params['F']:.4f}`")
        print(f"  - **Near Bound:**         `{cam_params['Near']:.4f}`")
        print(f"  - **Far Bound:**          `{cam_params['Far']:.4f}`")

        if i > 0 and is_baseline:
            print(f"  ##### Ratios to Camera 1 (in `{os.path.basename(file_path)}`):")
            if file_base_h != 0:
                print(
                    f"    - **H Ratio:**          `{(cam_params['H'] / file_base_h):.4f}`"
                )
            if file_base_w != 0:
                print(
                    f"    - **W Ratio:**          `{(cam_params['W'] / file_base_w):.4f}`"
                )
            if file_base_f != 0:
                print(
                    f"    - **F Ratio:**          `{(cam_params['F'] / file_base_f):.4f}`"
                )
            if file_base_near != 0:
                print(
                    f"    - **Near Ratio:**       `{(cam_params['Near'] / file_base_near):.4f}`"
                )
            if file_base_far != 0:
                print(
                    f"    - **Far Ratio:**        `{(cam_params['Far'] / file_base_far):.4f}`"
                )

        if (
            not is_baseline
            and baseline_info_for_ratios
            and i < baseline_info_for_ratios["num_cameras"]
        ):
            print(
                f"  ##### Ratios to Camera {cam_params['id']} (in Baseline: `{os.path.basename(baseline_info_for_ratios['file_path'])}`):"
            )
            base_cam_params = baseline_info_for_ratios["cameras_params"][i]
            if base_cam_params["H"] != 0:
                print(
                    f"    - **H Ratio:**          `{(cam_params['H'] / base_cam_params['H']):.4f}`"
                )
            if base_cam_params["W"] != 0:
                print(
                    f"    - **W Ratio:**          `{(cam_params['W'] / base_cam_params['W']):.4f}`"
                )
            if base_cam_params["F"] != 0:
                print(
                    f"    - **F Ratio:**          `{(cam_params['F'] / base_cam_params['F']):.4f}`"
                )
            if base_cam_params["Near"] != 0:
                print(
                    f"    - **Near Ratio:**       `{(cam_params['Near'] / base_cam_params['Near']):.4f}`"
                )
            if base_cam_params["Far"] != 0:
                print(
                    f"    - **Far Ratio:**        `{(cam_params['Far'] / base_cam_params['Far']):.4f}`"
                )

    if (
        not is_baseline
        and baseline_info_for_ratios
        and num_cameras != baseline_info_for_ratios["num_cameras"]
    ):
        print(
            f"\n**Warning:** Camera count mismatch in detailed report. Baseline (`{os.path.basename(baseline_info_for_ratios['file_path'])}`) has `{baseline_info_for_ratios['num_cameras']}` cameras, current file (`{os.path.basename(file_path)}`) has `{num_cameras}` cameras. Cross-file ratios shown for matching indices."
        )

    print(
        f"\n### Raw Pose Sample (First 3x4 matrix of Camera 1 in `{os.path.basename(file_path)}`)"
    )
    if num_cameras > 0:
        first_camera_pose_rt = data[0, :12].reshape(3, 4)
        print(f"```\n{first_camera_pose_rt}\n```")
    else:
        print("No camera data for raw pose sample.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes poses_bounds.npy files, providing an executive summary of aggregate differences and detailed reports. Output is in Markdown."
    )
    parser.add_argument(
        "file_paths",
        nargs="+",
        help="Path(s) to .npy file(s). First is baseline, others are compared to it.",
    )

    args = parser.parse_args()

    if not args.file_paths:
        print(
            "\n**Error:** No .npy files provided. Please specify at least one file path."
        )
    else:
        all_file_data = []
        print("\n# Pose File Analysis Report")  # Main Title for the whole report
        print("\nProcessing input files...")
        for fp in args.file_paths:
            data = extract_camera_data(fp)
            if data:
                all_file_data.append(data)
            else:
                # AI-generated comment: If a file fails to load, skip it for summaries and detailed reports.
                print(f"Skipping file `{fp}` due to loading errors.")

        if not all_file_data:
            print("\nNo valid files were processed. Exiting.")
        else:
            baseline_info = all_file_data[0]

            # AI-generated comment: Section for all executive summaries.
            if len(all_file_data) > 1:
                print("\n# Executive Summaries of Aggregate File Differences")
                for i in range(1, len(all_file_data)):
                    comparison_info = all_file_data[i]
                    summary = generate_executive_summary(baseline_info, comparison_info)
                    for line in summary:
                        print(line)

            # AI-generated comment: Section for all detailed reports.
            print("\n# Detailed File Reports")
            # Print baseline detailed report first
            generate_detailed_markdown_report(baseline_info, is_baseline=True)

            # Print comparison files detailed reports
            if len(all_file_data) > 1:
                for i in range(1, len(all_file_data)):
                    comparison_info = all_file_data[i]
                    generate_detailed_markdown_report(
                        comparison_info,
                        is_baseline=False,
                        baseline_info_for_ratios=baseline_info,
                    )

    print("\n## Analysis Complete")
