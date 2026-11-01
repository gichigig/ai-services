#!/usr/bin/env python3
"""
Dataset Compiler & Augmentation Engine for eCitizen Visual Action Model
Compiles the 18 labeled eCitizen BRS Visual Playbook steps into a multi-task
fine-tuning dataset for Qwen2.5-VL / Qwen2-VL Vision-Language Action Models.
"""

import os
import json
import random
from PIL import Image, ImageEnhance

# Normalized bounding boxes (ymin, xmin, ymax, xmax) on 0-1000 scale and click points (cx, cy)
# Grounded directly to the 1024x640 playbook screenshots
STEP_VISUAL_GROUNDINGS = {
    "step_00_login_checkpoint": {
        "box": [420, 360, 580, 640],
        "point": [500, 500],
        "target": "Sign In Form",
        "role": "form",
        "container": "One Log In Card",
        "action": "wait",
        "is_checkpoint": True,
        "handoff_trigger": "login",
        "user_action": "Enter eCitizen email or ID and password to sign in."
    },
    "step_01_dashboard": {
        "box": [220, 690, 270, 830],
        "point": [760, 245],
        "target": "Make Application",
        "role": "button",
        "container": "My Applications Card",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_02_select_brs_v2": {
        "box": [480, 580, 530, 680],
        "point": [630, 505],
        "target": "Apply now",
        "role": "button",
        "container": "Brs V2 Card",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_03_confirm_modal": {
        "box": [620, 590, 675, 715],
        "point": [652, 648],
        "target": "Apply Now »",
        "role": "button",
        "container": "Modal Notice: BRS V2",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_04_brsv2_sso_login": {
        "box": [510, 410, 570, 610],
        "point": [510, 540],
        "target": "Login with eCitizen",
        "role": "button",
        "container": "BRS V2 Login Card",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_05_account_switcher": {
        "box": [380, 360, 480, 640],
        "point": [500, 430],
        "target": "Personal Account",
        "role": "card_button",
        "container": "Select Account Card",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_06_oauth_consent": {
        "box": [660, 440, 715, 580],
        "point": [510, 688],
        "target": "Proceed",
        "role": "button",
        "container": "Authorization Consent Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_07_select_companies_registry": {
        "box": [560, 460, 615, 555],
        "point": [508, 588],
        "target": "Proceed",
        "role": "button",
        "container": "Companies Registry Services Card",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_08_apply_business_name": {
        "box": [365, 490, 415, 555],
        "point": [522, 390],
        "target": "Apply",
        "role": "button",
        "container": "Registration Of A Business Name",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_09_proposed_names": {
        "box": [820, 840, 880, 930],
        "point": [885, 850],
        "target": "Next",
        "role": "button",
        "container": "Proposed Names Step Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_10_applicant_details": {
        "box": [830, 840, 890, 930],
        "point": [885, 860],
        "target": "Next",
        "role": "button",
        "container": "Applicant Details Step Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_11_nature_of_business": {
        "box": [830, 840, 890, 930],
        "point": [885, 860],
        "target": "Next",
        "role": "button",
        "container": "Nature Of Business Step Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_12_registered_office_address": {
        "box": [830, 840, 890, 930],
        "point": [885, 860],
        "target": "Next",
        "role": "button",
        "container": "Registered Office Address Step Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_13_ownership_information": {
        "box": [830, 840, 890, 930],
        "point": [885, 860],
        "target": "Next",
        "role": "button",
        "container": "Ownership Information Step Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_14_application_documents": {
        "box": [470, 275, 540, 620],
        "point": [448, 505],
        "target": "Dully signed BN - 2 Form Upload",
        "role": "file_upload",
        "container": "Application Documents Step Form",
        "action": "wait",
        "is_checkpoint": True,
        "handoff_trigger": "file_upload",
        "user_action": "Please sign and upload the statutory BN-2 form. Autopilot will automatically resume once the document is attached."
    },
    "step_15_declaration": {
        "box": [595, 340, 645, 405],
        "point": [372, 620],
        "target": "Preview",
        "role": "button",
        "container": "Declaration Confirmation Form",
        "action": "click",
        "is_checkpoint": False,
        "handoff_trigger": None,
        "user_action": None
    },
    "step_16_application_review_complete": {
        "box": [920, 275, 970, 348],
        "point": [312, 945],
        "target": "Complete",
        "role": "button",
        "container": "Application Review Summary Form",
        "action": "click",
        "is_checkpoint": True,
        "handoff_trigger": "approval",
        "user_action": "Review application details. Click 'Complete' to file application with Registrar."
    },
    "step_17_payment_checkpoint": {
        "box": [540, 290, 880, 710],
        "point": [500, 710],
        "target": "Pesaflow Payment Options",
        "role": "payment_gateway",
        "container": "Pesaflow Checkout Portal",
        "action": "wait",
        "is_checkpoint": True,
        "handoff_trigger": "payment",
        "user_action": "Statutory invoice KES 925.00 generated on Pesaflow. Choose payment mode (Mpesa/Card) to authorize."
    }
}


def build_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    playbook_dir = os.path.join(base_dir, "playbooks", "ecitizen", "brs")
    dataset_dir = os.path.join(base_dir, "ai-service", "dataset")
    augmented_img_dir = os.path.join(dataset_dir, "images")
    os.makedirs(augmented_img_dir, exist_ok=True)

    manifest_path = os.path.join(playbook_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    steps = manifest.get("steps", [])
    print(f"Loaded {len(steps)} steps from manifest.")

    all_samples = []

    for step_idx, step_meta in enumerate(steps):
        step_id = step_meta.get("step_id")
        step_title = step_meta.get("step_title")
        img_filename = step_meta.get("guide_image")
        src_img_path = os.path.join(playbook_dir, img_filename)

        if not os.path.exists(src_img_path):
            print(f"Warning: Image {src_img_path} does not exist, skipping.")
            continue

        grounding = STEP_VISUAL_GROUNDINGS.get(step_id, {
            "box": [500, 500, 600, 600],
            "point": [550, 550],
            "target": step_meta.get("target", {}).get("label", "Target"),
            "role": step_meta.get("target", {}).get("role", "button"),
            "container": step_meta.get("target", {}).get("container", "Page"),
            "action": step_meta.get("target", {}).get("action", "click"),
            "is_checkpoint": False,
            "handoff_trigger": None,
            "user_action": None
        })

        # Load base image and create 3 variations (base, slightly brighter, slightly contrast-enhanced)
        with Image.open(src_img_path) as raw_im:
            base_im = raw_im.convert("RGB")
            w, h = base_im.size
            variations = [("base", base_im)]

            # Variation 1: Brightness shift
            enhancer_b = ImageEnhance.Brightness(base_im)
            im_bright = enhancer_b.enhance(1.08)
            variations.append(("bright", im_bright))

            # Variation 2: Contrast shift
            enhancer_c = ImageEnhance.Contrast(base_im)
            im_contrast = enhancer_c.enhance(1.06)
            variations.append(("contrast", im_contrast))

            for var_name, im_var in variations:
                dst_img_name = f"{step_id}_{var_name}.jpg"
                dst_img_path = os.path.join(augmented_img_dir, dst_img_name)
                im_var.save(dst_img_path, format="JPEG", quality=95)

                rel_img_path = os.path.relpath(dst_img_path, base_dir)

                box_str = f"({grounding['box'][0]},{grounding['box'][1]},{grounding['box'][2]},{grounding['box'][3]})"
                point_json = json.dumps({
                    "point": grounding["point"],
                    "target": grounding["target"],
                    "role": grounding["role"],
                    "container": grounding["container"],
                    "action": grounding["action"]
                })

                # Task 1: Step Identification
                all_samples.append({
                    "id": f"{step_id}_{var_name}_task1_step_id",
                    "step_id": step_id,
                    "image": rel_img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat eCitizen portal screen, step, and state is displayed in this image?"
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({
                                "step_id": step_id,
                                "step_title": step_title,
                                "service": "Business Registration Service (BRS)",
                                "is_checkpoint": grounding["is_checkpoint"],
                                "handoff_trigger": grounding["handoff_trigger"]
                            }, indent=2)
                        }
                    ]
                })

                # Task 2: Next Action Decision
                all_samples.append({
                    "id": f"{step_id}_{var_name}_task2_next_action",
                    "step_id": step_id,
                    "image": rel_img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nTask Goal: Register a business name on eCitizen Kenya. Analyze this screen and determine the next action."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({
                                "thought": f"Recognized {step_title}. Navigating forward toward business name registration.",
                                "action": grounding["action"],
                                "target": grounding["target"],
                                "container": grounding["container"],
                                "requires_user_approval": grounding["is_checkpoint"],
                                "reason": f"Execute {grounding['action']} on '{grounding['target']}' in '{grounding['container']}'."
                            }, indent=2)
                        }
                    ]
                })

                # Task 3: Visual Coordinate Grounding
                all_samples.append({
                    "id": f"{step_id}_{var_name}_task3_grounding",
                    "step_id": step_id,
                    "image": rel_img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nDetect and ground the target UI control for this step: \"{grounding['target']}\". Provide normalized bounding box coordinates [ymin, xmin, ymax, xmax] (0-1000) and click point [x, y]."
                        },
                        {
                            "from": "gpt",
                            "value": f"<|box_start|>{box_str}<|box_end|> {point_json}"
                        }
                    ]
                })

                # Task 4: Checkpoint & Safety Guard
                all_samples.append({
                    "id": f"{step_id}_{var_name}_task4_safety",
                    "step_id": step_id,
                    "image": rel_img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nExamine this screen for human-in-the-loop checkpoints, statutory signatures, or payment gates."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({
                                "is_checkpoint": grounding["is_checkpoint"],
                                "handoff_trigger": grounding["handoff_trigger"],
                                "action": "wait" if grounding["is_checkpoint"] else "proceed",
                                "user_action_required": grounding["user_action"],
                                "policy": "Statutory human interaction mandated." if grounding["is_checkpoint"] else "Autopilot may proceed automatically."
                            }, indent=2)
                        }
                    ]
                })

    # Shuffle with seed for deterministic train/val split
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.85)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_path = os.path.join(dataset_dir, "ecitizen_vlm_train.json")
    val_path = os.path.join(dataset_dir, "ecitizen_vlm_val.json")
    stats_path = os.path.join(dataset_dir, "dataset_stats.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "steps_covered": len(steps),
        "tasks_per_step": 4,
        "augmentations_per_image": 3,
        "format": "Qwen2.5-VL / Qwen2-VL Chat Format",
        "grounding_notation": "<|box_start|>(ymin,xmin,ymax,xmax)<|box_end|>"
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Successfully generated dataset:")
    print(f"- Total Samples: {len(all_samples)}")
    print(f"- Training Samples: {len(train_samples)} ({train_path})")
    print(f"- Validation Samples: {len(val_samples)} ({val_path})")
    print(f"- Stats: {stats_path}")


if __name__ == "__main__":
    build_dataset()
