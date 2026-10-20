'''RoboCasa eval client WITH rollout collection (LeRobot v2.1 self-distillation).

This is ``eval_robocasa_client_old.py`` plus four surgical hooks that record every
evaluated rollout (success AND failure) as a byte-compatible LeRobot v2.1 dataset
via ``rollout_collector.RolloutCollector``. The original evaluation behavior is
unchanged when ``--save_rollout`` is not passed.

Run on the RoboCasa simulator container (needs: robocasa, gymnasium, plus
numpy/pyarrow/imageio/PIL for the collector). The inference server
(serve_robocasa_policy.py) is unchanged and needs no knowledge of this.

Collect example:
  python eval_robocasa_client_rollout.py \
    --task_set target_atomic18 \
    --server_url http://127.0.0.1:7891 \
    --split pretrain \
    --num_trials 50 \
    --save_video_every 0 \
    --save_rollout \
    --rollout_root data/robocasa365_rollout
'''

import argparse
import collections
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Make rollout_collector importable regardless of CWD (expected next to this file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_collector import RolloutCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_ATOMIC18_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

TARGET_COMPOSITE_SEEN16_TASKS = [
    "StackBowlsCabinet",
    "PreSoakPan",
    "ScrubCuttingBoard",
    "WashLettuce",
    "RinseSinkBasin",
    "KettleBoiling",
    "LoadDishwasher",
    "StoreLeftoversInBowl",
    "SetUpCuttingStation",
    "StirVegetables",
    "SteamInMicrowave",
    "SearingMeat",
    "PackIdenticalLunches",
    "PrepareCoffee",
    "DeliverStraw",
    "GetToastedBread",
]

TARGET_COMPOSITE_UNSEEN16_TASKS = [
    "RecycleBottlesByType",
    "WaffleReheat",
    "ArrangeBreadBasket",
    "WeighIngredients",
    "BreadSelection",
    "CuttingToolSelection",
    "GarnishPancake",
    "ArrangeTea",
    "WashFruitColander",
    "MakeIceLemonade",
    "PortionHotDogs",
    "PanTransfer",
    "HeatKebabSandwich",
    "CategorizeCondiments",
    "SeparateFreezerRack",
    "GatherTableware",
]

PRETRAIN_ATOMIC1 = [
    "AdjustToasterOvenTemperature", "AdjustWaterTemperature", "CheesyBread", "CloseBlenderLid",
    "CloseCabinet", "CloseDishwasher", "CloseDrawer", "CloseElectricKettleLid",
    "CloseFridge", "CloseFridgeDrawer", "CloseMicrowave", "CloseOven",
    "CloseStandMixerHead", "CloseToasterOvenDoor", "CoffeeServeMug", "CoffeeSetupMug",
    "LowerHeat", "MakeIcedCoffee", "NavigateKitchen", "OpenBlenderLid",
    "OpenCabinet", "OpenDishwasher", "OpenDrawer", "OpenElectricKettleLid",
    "OpenFridge", "OpenFridgeDrawer", "OpenMicrowave", "OpenOven",
    "OpenStandMixerHead", "OpenToasterOvenDoor", "PackDessert", "PickPlaceCabinetToCounter",
]

PRETRAIN_ATOMIC2 = [
    "PickPlaceCounterToBlender", "PickPlaceCounterToCabinet", "PickPlaceCounterToDrawer", "PickPlaceCounterToMicrowave",
    "PickPlaceCounterToOven", "PickPlaceCounterToSink", "PickPlaceCounterToStandMixer", "PickPlaceCounterToStove",
    "PickPlaceCounterToToasterOven", "PickPlaceDrawerToCounter", "PickPlaceFridgeDrawerToShelf", "PickPlaceFridgeShelfToDrawer",
    "PickPlaceMicrowaveToCounter", "PickPlaceSinkToCounter", "PickPlaceStoveToCounter", "PickPlaceToasterOvenToCounter",
    "PickPlaceToasterToCounter", "PreheatOven", "SlideDishwasherRack", "SlideOvenRack",
    "SlideToasterOvenRack", "StartCoffeeMachine", "TurnOffMicrowave", "TurnOffSinkFaucet",
    "TurnOffStove", "TurnOnBlender", "TurnOnElectricKettle", "TurnOnMicrowave",
    "TurnOnSinkFaucet", "TurnOnStove", "TurnOnToaster", "TurnOnToasterOven", "TurnSinkSpout",
]

PRETRAIN_COMPOSITE1 = [
    "AddIceCubes", "AddLemonToFish", "AddMarshmallow", "AddSugarCubes",
    "AddSweetener", "AdjustHeat", "AfterwashSorting", "AirDryFruit",
    "AlcoholServingPrep", "AlignSilverware", "ArrangeBreadBowl", "ArrangeBuffetDessert",
    "ArrangeDrinkware", "ArrangeTeaAccompaniments", "ArrangeUtensilsByType", "ArrangeVegetables",
    "AssembleCookingArray", "BalancedMealPrep", "BeverageOrganization", "BeverageSorting",
]

PRETRAIN_COMPOSITE2 = [    
    "BlendIngredients", "BlendMarinade", "BowlAndCup", "BreadAndCheese",
    "BreadSetupSlicing", "BuildAppetizerPlate", "ButterOnPan", "CandleCleanup",
    "CerealAndBowl", "ChooseMeasuringCup", "ChooseRipeFruit", "CleanBoard",
    "CleanMicrowave", "ClearClutter", "ClearCuttingBoard", "ClearFoodWaste",
    "ClearFreezer", "ClearReceptaclesForCleaning", "ClearSink", "ClearSinkArea",
]


PRETRAIN_COMPOSITE3= [
    "ClearSinkSpace", "CollectWashingSupplies", "ColorfulSalsa", "CondimentCollection",
    "CookieDoughPrep", "CoolBakedCake", "CoolKettle", "CreateChildFriendlyFridge",
    "CupcakeCleanup", "CutBuffetPizza", "DateNight", "DefrostByCategory",
    "DeliverBrewedCoffee", "DeliverStraw", "DessertAssembly", "DessertUpgrade",
    "DisplayMeatVariety", "DistributeChicken", "DivideBasins", "DivideBuffetTrays",
    "DrainVeggies", "DrinkwareConsolidation", "DryDishes", "DryDrinkware",
]

PRETRAIN_COMPOSITE4= [
    "DumpLeftovers", "FillBlenderJug", "FillKettle", "FilterMicrowavableItem",
    "FoodCleanup", "FreezeBottledWaters", "FreezeCookedFood", "FreezeIceTray",
    "FreshProduceOrganization", "FryingPanAdjustment", "GarnishCupcake", "GatherCuttingTools",
    "GatherMarinadeIngredients", "GatherVegetables", "GetToastedBread", "HeatMug",
    "HeatMultipleWater", "HotDogSetup", "JuiceFruitReamer", "KettleBoiling",
]

PRETRAIN_COMPOSITE5 = [
    "LemonSeasoningFish", "LineUpCondiments", "LoadCondimentsInFridge", "LoadDishwasher",
    "LoadFridgeByType", "LoadFridgeFifo", "LoadPreparedFood", "MakeFruitBowl",
    "MakeLoadedPotato", "MatchCupAndDrink", "MaximizeFreezerSpace", "MealPrepStaging",
    "MeatSkewerAssembly", "MeatTransfer", "MicrowaveCorrectMeal", "MicrowaveDefrostMeat",
    "MicrowaveThawing", "MicrowaveThawingFridge", "MixCakeFrosting", "MixedFruitPlatter",
]

PRETRAIN_COMPOSITE6 = [
    "MoveFreezerToFridge", "MoveFridgeToFreezer", "MoveToCounter", "MoveToFreezerDrawer",
    "MultistepSteaming", "OrganizeBakingIngredients", "OrganizeCleaningSupplies", "OrganizeCoffeeCondiments",
    "OrganizeCondiments", "OrganizeMetallicUtensils", "OrganizeMugsByHandle", "OrganizeVegetables",
    "OvenBroilFish", "PackFoodByTemp", "PackFruitContainer", "PackIdenticalLunches",
    "PastryDisplay", "PlaceBeveragesTogether", "PlaceDishesBySink", "PlaceEqualIceCubes",
]


PRETRAIN_COMPOSITE7 = [
    "PlaceFoodInBowls", "PlaceIceInCup", "PlaceMeatInMarinade", "PlaceMicrowaveSafeItem",
    "PlaceStraw", "PlaceVegetablesEvenly", "PlaceVeggiesInDrawer", "PlateSteakMeal",
    "PlateStoreDinner", "PortionFruitBowl", "PortionInTupperware", "PortionOnSize",
    "PortionYogurt", "PreRinseStation", "PreSoakPan", "PreheatPot",
    "PrepForSanitizing", "PrepForTenderizing", "PrepFridgeForCleaning", "PrepMarinatingMeat",
]

PRETRAIN_COMPOSITE8 = [
    "PrepSinkForCleaning", "PrepareBroilingStation", "PrepareCheeseStation", "PrepareCocktailStation",
    "PrepareCoffee", "PrepareDishwasher", "PrepareDrinkStation", "PrepareSausageCheese",
    "PrepareSmoothie", "PrepareSoupServing", "PrepareStoringLeftovers", "PrepareToast",
    "PressChicken", "PrewashFoodAssembly", "PrewashFoodSorting", "QuickThaw",
    "RearrangeFridgeItems", "RecycleBottlesBySize", "RecycleSodaCans", "RecycleStackedYogurt",
]


PRETRAIN_COMPOSITE9 = [
    "RefillCondimentStation", "ReheatMeal", "RemoveBroiledFish", "RemoveCuttingBoardItems",
    "ReorganizeFrozenVegetables", "ResetCabinetDoors", "RestockBowls", "RestockCannedFood",
    "RestockPantry", "RestockSinkSupplies", "RetrieveIceTray", "RetrieveMeat",
    "ReturnHeatedFood", "ReturnWashingSupplies", "RinseBowls", "RinseCuttingBoard",
    "RinseSinkBasin", "RotatePan", "SanitizePrepCuttingBoard", "ScalePortioning",
]

PRETRAIN_COMPOSITE10 = [
    "ScrubBowl", "ScrubCuttingBoard", "SearingMeat", "SeasoningSpiceSetup",
    "SeasoningSteak", "ServeSteak", "ServeTea", "ServeWarmCroissant",
    "SetBowlsForSoup", "SetUpCuttingStation", "SetUpSpiceStation", "SetupBowls",
    "SetupButterPlate", "SetupFruitBowl", "SetupFrying", "SetupSodaBowl",
]

PRETRAIN_COMPOSITE11 = [
    "SetupWineGlasses", "ShakePan", "SimmeringSauce", "SizeSorting",
    "SnackSorting", "SoakSponge", "SortingCleanup", "SpicyMarinade",
    "StackBowlsCabinet", "StackBowlsInSink", "StackCans", "StartElectricKettle",
    "SteamInMicrowave", "StirVegetables", "StockingBreakfastFoods", "StoreDumplings",
    "StoreLeftoversByType", "StoreLeftoversInBowl", "StrainerSetup", "SweetSavoryToastSetup",
]

PRETRAIN_COMPOSITE12 = [
    "SweetenCoffee", "SweetenHotChocolate", "ThawInSink", "TiltPan",
    "ToastBagel", "ToastBaguette", "ToastOnCorrectRack", "TongBuffetSetup",
    "TransportCookware", "TurnOffSimmeredSauceHeat", "VeggieDipPrep", "WarmCroissant",
    "WashFish", "WashLettuce", "YogurtDelightPrep",
]




PRETRAIN_MISSING = [
    "DrinkwareConsolidation","FillBlenderJug",
    "FreezeBottledWaters","FryingPanAdjustment",
    "GatherVegetables","HotDogSetup",
]

PRETRAIN_MISSING2 = [
    "DryDishes",
]

PRETRAIN_MISSING3 = [
    "RecycleSodaCans",
]

TARGET_ALL50_TASKS = (
    TARGET_ATOMIC18_TASKS
    + TARGET_COMPOSITE_SEEN16_TASKS
    + TARGET_COMPOSITE_UNSEEN16_TASKS
)


PRETRAIN_ATOMIC = (
    PRETRAIN_ATOMIC1
    + PRETRAIN_ATOMIC2
)

PRETRAIN_COMPOSITE = (
    PRETRAIN_COMPOSITE1
    + PRETRAIN_COMPOSITE2
    + PRETRAIN_COMPOSITE3
)

PRETRAIN_ALL300_TASKS = (
    PRETRAIN_ATOMIC
    + PRETRAIN_COMPOSITE
)


TASK_SETS = {
    "target_atomic18": TARGET_ATOMIC18_TASKS,
    "target_atomic": TARGET_ATOMIC18_TASKS,
    "target_composite_seen16": TARGET_COMPOSITE_SEEN16_TASKS,
    "target_composite_unseen16": TARGET_COMPOSITE_UNSEEN16_TASKS,
    "target_all50": TARGET_ALL50_TASKS,
    "pretrain_atomic":PRETRAIN_ATOMIC,
    "pretrain_atomic1":PRETRAIN_ATOMIC1,
    "pretrain_atomic2":PRETRAIN_ATOMIC2,
    "pretrain_composite":PRETRAIN_COMPOSITE,
    "pretrain_composite1":PRETRAIN_COMPOSITE1,
    "pretrain_composite2":PRETRAIN_COMPOSITE2,
    "pretrain_composite3":PRETRAIN_COMPOSITE3,
    "pretrain_composite4":PRETRAIN_COMPOSITE4,
    "pretrain_composite5":PRETRAIN_COMPOSITE5,
    "pretrain_composite6":PRETRAIN_COMPOSITE6,
    "pretrain_composite7":PRETRAIN_COMPOSITE7,
    "pretrain_composite8":PRETRAIN_COMPOSITE8,
    "pretrain_composite9":PRETRAIN_COMPOSITE9,
    "pretrain_composite10":PRETRAIN_COMPOSITE10,
    "pretrain_composite11":PRETRAIN_COMPOSITE11,
    "pretrain_composite12":PRETRAIN_COMPOSITE12,
    "pretrain300":PRETRAIN_ALL300_TASKS,
    "pretrain_missing":PRETRAIN_MISSING,
    "pretrain_missing2":PRETRAIN_MISSING2,
    "pretrain_missing3":PRETRAIN_MISSING3,
}

TASK_SUBSET_BY_NAME = {
    **{task: "atomic18" for task in TARGET_ATOMIC18_TASKS},
    **{task: "composite_seen16" for task in TARGET_COMPOSITE_SEEN16_TASKS},
    **{task: "composite_unseen16" for task in TARGET_COMPOSITE_UNSEEN16_TASKS},
}


class DM05InferenceClient:
    """HTTP client for remote DM05 inference service."""

    def __init__(self, server_url, robot_type="panda_omron"):
        self.server_url = server_url.rstrip("/")
        self.process_url = f"{self.server_url}/process_frame"
        self.robot_type = robot_type

        logger.info(f"Connecting to inference server at {self.server_url}...")
        try:
            requests.get(self.server_url, timeout=5)
            logger.info("Server is reachable!")
        except Exception as e:
            logger.warning(
                f"Cannot reach server at {self.server_url}: {e}\n"
                "Make sure the inference service is running on the dexbotic server."
            )

    def predict(self, images, state, prompt):
        image_files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            image_files.append(("image", (f"image_{i}.png", buf, "image/png")))

        form_data = {
            "text": prompt,
            "states": json.dumps(state.tolist()),
            "robot_type": self.robot_type,
        }

        try:
            resp = requests.post(
                self.process_url,
                data=form_data,
                files=image_files,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            if "response" not in result:
                raise RuntimeError(f"Malformed inference response: {result}")

            actions = np.array(result["response"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise RuntimeError(f"Invalid action chunk shape: {actions.shape}")
            return actions
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            raise


def _compact_action_to_native(action):
    """Convert compact 11D PandaOmron action to native 12D if needed."""
    action = np.asarray(action, dtype=np.float64).flatten()
    if action.shape[0] == 12:
        return action
    if action.shape[0] != 11:
        raise ValueError(f"Expected PandaOmron action dim 11 or 12, got {action.shape}")

    native = np.zeros(12, dtype=np.float64)
    native[0:3] = action[0:3]
    native[4] = action[3]
    native[5:8] = action[4:7]
    native[8:11] = action[7:10]
    native[11] = action[10]
    return native


def model_action_to_env_action(model_action):
    """Convert PandaOmron model action to RoboCasa env action dict.

    Hard mode constraint:
      - control_mode >= 0  -> base mode, zero EEF branch
      - control_mode <  0  -> EEF mode, zero base branch
    """
    action = _compact_action_to_native(model_action)

    base_motion = action[0:4].copy()
    control_mode = float(action[4])
    eef_pos = action[5:8].copy()
    eef_rot = action[8:11].copy()
    gripper = action[11:12].copy()

    # if control_mode >= 0:
    #     control_mode = 1.0
    #     eef_pos[:] = 0.0
    #     eef_rot[:] = 0.0
    # else:
    #     control_mode = -1.0
    #     base_motion[:] = 0.0

    # [-1,1] -> [0,1]，因为robocasa365环境内部有阈值判断，并且对于夹爪阈值是0.5
    control_mode_env = (control_mode + 1.0) / 2.0
    gripper_env = (gripper + 1.0) / 2.0
    env_action = {
        "action.base_motion": base_motion,
        "action.control_mode": np.array([control_mode_env], dtype=np.float64),
        "action.end_effector_position": eef_pos,
        "action.end_effector_rotation": eef_rot,
        "action.gripper_close": gripper_env,
    }
    return env_action




def _get_obs_array(obs, candidates, expected_dim=None):
    for key in candidates:
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float32)
            if expected_dim is not None and arr.shape[-1] != expected_dim:
                raise ValueError(
                    f"Observation key {key!r} has dim {arr.shape[-1]}, expected {expected_dim}"
                )
            return arr
    raise KeyError(f"None of the observation keys exist: {candidates}")


def get_state_from_obs(obs):
    """Build compact 14D PandaOmron state from RoboCasa observation."""
    base_pos = _get_obs_array(
        obs,
        ["state.base_position", "robot_state.base_position"],
        expected_dim=3,
    )
    base_rot = _get_obs_array(
        obs,
        ["state.base_rotation", "robot_state.base_rotation"],
        expected_dim=4,
    )
    eef_pos = _get_obs_array(
        obs,
        ["state.end_effector_position_relative"],
        expected_dim=3,
    )
    eef_rot = _get_obs_array(
        obs,
        ["state.end_effector_rotation_relative"],
        expected_dim=4,
    )
    gripper = _get_obs_array(
        obs,
        ["state.gripper_qpos"],
        expected_dim=2,
    )

    state = np.concatenate(
        [
            base_pos,
            base_rot[2:4],
            eef_pos,
            eef_rot,
            gripper,
        ],
        axis=0,
    ).astype(np.float32)

    if state.shape[0] != 14:
        raise ValueError(f"Expected 14D compact PandaOmron state, got {state.shape}")
    return state


def get_images_from_obs(obs):
    camera_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    images = []
    for key in camera_keys:
        img_array = obs[key]
        img = Image.fromarray(np.ascontiguousarray(img_array)).convert("RGB")
        images.append(img)
    return images


def should_save_video(episode_idx, save_video_every):
    if save_video_every <= 0:
        return False
    return ((episode_idx + 1) % save_video_every) == 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _episodes_meta_path(collector, task_name):
    return Path(collector.out_root) / task_name / collector.subdir / "meta" / "episodes.jsonl"


def _load_existing_episode_records(collector, task_name):
    if collector is None:
        return []
    path = _episodes_meta_path(collector, task_name)
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skip malformed existing rollout meta line: %s:%d (%s)",
                    path,
                    line_no,
                    exc,
                )
    return records


def _finalize_rollout_episode(collector, ep_buf, task_name, done, env_seed):
    if ep_buf is None or len(ep_buf) == 0:
        return None

    try:
        saved_idx = collector.finish_episode(
            ep_buf,
            success=bool(done),
            env_seed=env_seed,
            min_length=1,
        )
        logger.info(
            "  [rollout] task=%s saved episode_index=%s success=%s frames=%d",
            task_name,
            saved_idx,
            bool(done),
            len(ep_buf),
        )
        return saved_idx
    except PermissionError as exc:
        logger.error(
            "  [rollout] permission denied while saving task=%s env_seed=%s: %s",
            task_name,
            env_seed,
            exc,
        )
        logger.error(
            "  [rollout] Check write permission / ownership under rollout_root for this task."
        )
        raise


def evaluate(
    task_name,
    server_url,
    num_trials=10,
    split="target",
    seed=42,
    video_dir="eval_videos",
    run_name=None,
    replan_steps=5,
    max_steps=None,
    save_video_every=1,
    robot_type="panda_omron",
    collector=None,
    resume=True,
):
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    np.random.seed(seed)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(video_dir, run_name, task_name)
    ensure_dir(output_dir)
    episode_results_path = os.path.join(output_dir, "episode_results.jsonl")

    existing_records = _load_existing_episode_records(collector, task_name) if resume else []
    existing_count = min(len(existing_records), num_trials)
    existing_successes = sum(int(bool(rec.get("success", False))) for rec in existing_records[:existing_count])

    if existing_count > 0:
        logger.info(
            "[resume] %s existing episodes on disk: %d/%d",
            task_name,
            existing_count,
            num_trials,
        )

    if existing_count >= num_trials:
        final_stats = {
            "task": task_name,
            "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
            "split": split,
            "num_episodes": existing_count,
            "num_successes": existing_successes,
            "success_rate": existing_successes / existing_count if existing_count > 0 else 0.0,
            "server_url": server_url,
            "robot_type": robot_type,
            "seed": seed,
            "replan_steps": replan_steps,
            "max_steps": max_steps,
            "save_video_every": save_video_every,
            "run_name": run_name,
            "episode_results_path": episode_results_path,
            "resumed_from_existing": existing_count,
        }
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info("[resume] %s already complete, skip env rollout.", task_name)
        return final_stats

    client = DM05InferenceClient(server_url, robot_type=robot_type)

    logger.info(f"Creating environment: robocasa/{task_name}, split={split}")
    env = gym.make(
        f"robocasa/{task_name}",
        split=split,
        seed=seed,
    )

    if max_steps is None:
        max_steps = get_task_horizon(task_name)
        max_steps = int(max_steps * 1.2) 
    logger.info(f"Max steps per episode: {max_steps}")

    total_successes = existing_successes
    total_episodes = existing_count

    for episode_idx in tqdm(range(existing_count, num_trials), desc=f"Evaluating {task_name}"):
        record_video = should_save_video(episode_idx, save_video_every)
        replay_images = [] if record_video else None

        env_seed = seed + episode_idx
        obs, info = env.reset(seed=env_seed)
        task_lang = str(obs.get("annotation.human.task_description", f"Task: {task_name}"))
        logger.info(f"\nEpisode {episode_idx + 1}/{num_trials}: {task_lang}")

        # --- rollout hook (STEP 2): begin an episode buffer ---
        ep_buf = None
        if collector is not None:
            ep_buf = collector.start_episode(
                task_name=task_name,
                task_description=task_lang,
            )

        action_plan = collections.deque()
        done = False
        terminated = False
        truncated = False
        t = 0

        for t in range(max_steps):
            if not action_plan:
                images = get_images_from_obs(obs)
                state = get_state_from_obs(obs)

                t0 = time.time()
                action_chunk = client.predict(images, state, task_lang)
                dt = time.time() - t0

                if t == 0:
                    logger.info(f"  First inference took {dt:.2f}s, got {len(action_chunk)} actions")

                steps_to_use = min(replan_steps, len(action_chunk))
                for i in range(steps_to_use):
                    action_plan.append(action_chunk[i])

            raw_action = action_plan.popleft()

            # --- rollout hook (STEP 3): capture obs + executed action BEFORE stepping ---
            if ep_buf is not None:
                rec_images = get_images_from_obs(obs)   # 3x PIL, client order
                rec_state = get_state_from_obs(obs)     # np [14] compact

            env_action = model_action_to_env_action(raw_action)
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = info.get("success", False)

            if ep_buf is not None:
                # raw_action is the compact 11D action we executed; the collector
                # expands it to native 12D internally.
                ep_buf.add_step(
                    images_client_order=rec_images,
                    state_compact_14d=rec_state,
                    action_compact_11d=raw_action,
                    reward=float(reward),
                    done=bool(done),
                    video_size=(256, 256),
                )

            if record_video and (t % 2 == 0 or done or t == max_steps - 1):
                try:
                    render_img = env.render()
                    if render_img is not None:
                        replay_images.append(np.ascontiguousarray(render_img))
                except Exception:
                    pass

            if done:
                logger.info(f"  ✅ Success at step {t}!")
                break

            if terminated or truncated:
                logger.info(f"  Episode terminated early at step {t}")
                break

        if not done:
            logger.info(f"  ❌ Failed (reached max steps {max_steps})")

        # --- rollout hook (STEP 4): finalize the episode (success AND failure) ---
        _finalize_rollout_episode(
            collector=collector,
            ep_buf=ep_buf,
            task_name=task_name,
            done=done,
            env_seed=env_seed,
        )

        total_successes += int(done)
        total_episodes += 1

        if record_video and replay_images:
            suffix = "success" if done else "failure"
            video_path = os.path.join(output_dir, f"episode_{episode_idx + 1:03d}_{suffix}.mp4")
            imageio.mimwrite(video_path, replay_images, fps=20)

        episode_record = {
            "episode_idx": episode_idx,
            "task": task_name,
            "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
            "split": split,
            "seed": env_seed,
            "success": bool(done),
            "success_int": int(done),
            "steps_executed": t + 1,
            "max_steps": max_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "replan_steps": replan_steps,
            "robot_type": robot_type,
            "server_url": server_url,
            "run_name": run_name,
        }
        if record_video and replay_images:
            episode_record["video_path"] = video_path
        _append_jsonl(episode_results_path, episode_record)

        success_rate = total_successes / total_episodes * 100.0
        logger.info(
            f"  Progress: {total_episodes}/{num_trials}, "
            f"Success: {total_successes}/{total_episodes} ({success_rate:.1f}%)"
        )

    final_stats = {
        "task": task_name,
        "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
        "split": split,
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
        "server_url": server_url,
        "robot_type": robot_type,
        "seed": seed,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
        "save_video_every": save_video_every,
        "run_name": run_name,
        "episode_results_path": episode_results_path,
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Final Results for {task_name}:")
    logger.info(f"  Success Rate: {final_stats['success_rate'] * 100:.1f}%")
    logger.info(f"  ({total_successes}/{total_episodes} episodes)")
    logger.info(f"  Outputs saved to: {output_dir}")
    logger.info(f"  Episode records saved to: {episode_results_path}")
    logger.info(f"  Stats saved to: {stats_path}")
    logger.info("=" * 60)

    env.close()
    return final_stats


def evaluate_task_set(
    task_names,
    server_url,
    num_trials=50,
    split="target",
    seed=42,
    video_dir="eval_videos",
    run_name=None,
    replan_steps=5,
    max_steps=None,
    save_video_every=10,
    task_set_name="custom",
    robot_type="panda_omron",
    collector=None,
    resume=True,
):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    ensure_dir(os.path.join(video_dir, run_name))

    all_results = []
    for task_idx, task_name in enumerate(task_names):
        logger.info(
            f"\n{'#' * 70}\n"
            f"Task {task_idx + 1}/{len(task_names)}: {task_name}\n"
            f"{'#' * 70}"
        )

        task_seed = seed + task_idx * 1000
        stats = evaluate(
            task_name=task_name,
            server_url=server_url,
            num_trials=num_trials,
            split=split,
            seed=task_seed,
            video_dir=video_dir,
            run_name=run_name,
            replan_steps=replan_steps,
            max_steps=max_steps,
            save_video_every=save_video_every,
            robot_type=robot_type,
            collector=collector,
            resume=resume,
        )
        stats["task_subset"] = TASK_SUBSET_BY_NAME.get(task_name, "custom")
        all_results.append(stats)

    overall_success_rate = (
        float(np.mean([x["success_rate"] for x in all_results])) if all_results else 0.0
    )

    subset_summaries = {}
    subset_names = sorted({x["task_subset"] for x in all_results})
    for subset_name in subset_names:
        subset_results = [x for x in all_results if x["task_subset"] == subset_name]
        subset_summaries[subset_name] = {
            "num_tasks": len(subset_results),
            "overall_success_rate": (
                float(np.mean([x["success_rate"] for x in subset_results]))
                if subset_results else 0.0
            ),
            "tasks": [x["task"] for x in subset_results],
        }

    summary = {
        "task_set_name": task_set_name,
        "split": split,
        "num_tasks": len(task_names),
        "num_trials_per_task": num_trials,
        "overall_success_rate": overall_success_rate,
        "subset_summaries": subset_summaries,
        "server_url": server_url,
        "robot_type": robot_type,
        "seed": seed,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
        "save_video_every": save_video_every,
        "run_name": run_name,
        "tasks": all_results,
    }

    summary_path = os.path.join(video_dir, run_name, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"Task set finished: {task_set_name}")
    logger.info(f"Overall success rate: {overall_success_rate * 100:.1f}%")
    for subset_name, subset_summary in subset_summaries.items():
        logger.info(
            f"  {subset_name}: {subset_summary['overall_success_rate'] * 100:.1f}% "
            f"over {subset_summary['num_tasks']} tasks"
        )
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)

    return summary


def _parse_task_list(task_arg: str) -> list:
    tasks = [x.strip() for x in task_arg.split(",") if x.strip()]
    if not tasks:
        raise ValueError("--tasks is provided but no valid task names were parsed")
    duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
    if duplicates:
        raise ValueError(f"Duplicated task names in --tasks: {duplicates}")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DM05 model on RoboCasa (client mode) + optional rollout collection"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single task name (e.g., OpenDrawer, CloseFridge)",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        default=None,
        choices=sorted(TASK_SETS.keys()),
        help="Predefined task set name",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task list, e.g. OpenDrawer,CloseFridge,TurnOnMicrowave",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:7891",
        help="DM05 inference server URL",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="panda_omron",
        help="Robot type/profile passed to the DM05 inference server",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of evaluation episodes per task",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["target", "pretrain"],
        help="Environment split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="eval_videos",
        help="Directory to save evaluation videos and stats",
    )
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=5,
        help="Steps to execute before re-querying model",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (default: task-specific)",
    )
    parser.add_argument(
        "--save_video_every",
        type=int,
        default=1,
        help="Save one video every N episodes; 0 means save no videos",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name; default is timestamp",
    )
    # --- rollout collection args (STEP 1) ---
    parser.add_argument(
        "--save_rollout",
        action="store_true",
        help="Record rollouts (success AND failure) as LeRobot v2.1 datasets.",
    )
    parser.add_argument(
        "--rollout_root",
        type=str,
        default="data/robocasa365_rollout",
        help="Output root for rollout LeRobot datasets (one dir per task).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable auto-resume from existing rollout_root/<Task>/lerobot/meta/episodes.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    provided_modes = [
        args.task is not None,
        args.tasks is not None,
        args.task_set is not None,
    ]
    if sum(provided_modes) != 1:
        raise ValueError("Provide exactly one of --task, --tasks, or --task_set")

    # --- rollout hook (STEP 1): build one collector for the whole run ---
    collector = None
    if args.save_rollout:
        collector = RolloutCollector(
            out_root=args.rollout_root,
            fps=20,                    # RoboCasa365 native fps (info.json)
            video_size=(256, 256),     # native per-camera size
            provenance={
                "server_url": args.server_url,
                "split": args.split,
                "seed": args.seed,
                "replan_steps": args.replan_steps,
                "robot_type": args.robot_type,
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "resume": not args.no_resume,
            },
        )
        logger.info("Rollout collection ENABLED -> %s", args.rollout_root)

    if args.task is not None:
        evaluate(
            task_name=args.task,
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            robot_type=args.robot_type,
            collector=collector,
            resume=not args.no_resume,
        )
    elif args.tasks is not None:
        evaluate_task_set(
            task_names=_parse_task_list(args.tasks),
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            task_set_name="custom_tasks",
            robot_type=args.robot_type,
            collector=collector,
            resume=not args.no_resume,
        )
    else:
        evaluate_task_set(
            task_names=TASK_SETS[args.task_set],
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            task_set_name=args.task_set,
            robot_type=args.robot_type,
            collector=collector,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()

