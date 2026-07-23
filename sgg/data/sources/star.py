from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..io import load_image_tensor


BOX_SCALE = 2000
DEFAULT_FIXED_TRAIN_INDEX = [352, 1140, 781, 734, 138, 942, 319, 947, 461, 1187, 783, 714, 1078, 1117, 1193, 1047, 1113, 1108, 956, 1136, 987, 948, 560, 295, 835, 458, 444, 152, 326, 1016, 843, 278, 269, 462, 474, 137, 1037, 874, 270, 968, 1185, 503, 694, 997, 297, 616, 219, 387, 1195, 1197, 161, 840, 99, 713, 607, 431, 1036, 810, 403, 672, 992, 1133, 553, 1231, 634, 969, 554, 1121, 97, 93, 402, 1045, 1202, 925, 436, 375, 580, 282, 215, 768, 496, 647, 96, 472, 411, 735, 928, 718, 750, 746, 1056, 250, 1219, 1039, 902, 351, 1035, 1046, 1034, 490, 155, 234, 965, 88, 299, 678, 98, 487, 876, 192, 1033, 1073, 1253, 418, 1272, 144, 1070, 497, 499, 638, 575, 1040, 291, 854, 578, 1223, 1042, 917, 954, 376, 410, 257, 769, 432, 87, 752, 1218, 821, 799, 633, 686, 24, 89, 43, 434, 271, 1051, 1166, 571, 708, 715, 405, 908, 124, 583, 635, 48, 782, 1190, 1067, 511, 423, 59, 829, 324, 1247, 421, 627, 381, 858, 1260, 364, 559, 368, 941, 657, 316, 313, 33, 816, 170, 393, 332, 813, 915, 1212, 913, 592, 819, 609, 494, 54, 101, 919, 255, 848, 812, 171, 356, 1097, 443, 424, 60, 665, 1012, 391, 757, 709, 169, 762, 493, 955, 727, 1142, 429, 654, 624, 153, 392, 446, 615, 427, 1096, 515, 775, 682, 996, 388, 576, 287, 286, 83, 629, 832, 646, 979, 1205, 881, 378, 1264, 645, 516, 106, 359, 1132, 262, 1144, 475, 274, 1244, 339, 435, 618, 1220, 1111, 145, 1227, 1221, 42, 846, 766, 568, 641, 921, 814, 304, 864, 483, 1110, 442, 1116, 412, 1158, 437, 139, 510, 147, 1058, 450, 1269, 1137, 249, 127, 310, 828, 801, 690, 972, 1204, 826, 946, 1165, 166, 25, 92, 983, 502, 1225, 1168, 1206, 414, 300, 619, 631, 336, 1189, 1245, 441, 251, 877, 1094, 770, 151, 401, 974, 140, 1064, 808, 459, 964, 834, 1248, 522, 1055, 374, 679, 361, 23, 796, 1228, 594, 579, 512, 79, 659, 253, 358, 85, 649, 221, 1242, 628, 484, 1072, 588, 593, 1109, 922, 1145, 1066, 537, 501, 818, 860, 802, 107, 273, 765, 971, 457, 491, 100, 447, 863, 1182, 1211, 382, 321, 705, 544, 266, 505, 980, 1159, 1170, 985, 489, 284, 1075, 1048, 850, 867, 891, 482, 428, 967, 47, 728, 178, 582, 958, 1240, 261, 685, 349, 1243, 780, 803, 732, 479, 900, 150, 181, 892, 384, 202, 477, 856, 355, 298, 22, 1246, 385, 725, 492, 513, 1217, 70, 1014, 82, 148, 149, 301, 55, 1157, 622, 620, 506, 1258, 1038, 551, 1041, 1149, 338, 589, 37, 471, 561, 1059, 476, 790, 625, 320, 478, 509, 792, 500, 683, 702, 448, 644, 331, 240, 590, 556, 426, 1201, 1259, 495, 102, 674, 488, 377, 1226, 789, 383, 562, 211, 1122, 445, 1252, 263, 861, 651, 610, 105, 227, 658, 570, 422, 1169, 699, 1148, 878, 668, 289, 49, 1135, 259, 416, 630, 951, 275, 684, 930, 182, 572, 1254, 62, 653, 817, 110, 774, 517, 1203, 538, 940, 130, 470, 587, 640, 128, 1, 1210, 953, 1120, 648, 90, 94, 1134, 1268, 486, 1208, 667, 932, 317, 1229, 11, 541, 777, 784, 844, 146, 330, 420, 567, 542, 737, 585, 529, 1052, 1054, 51, 1261, 173, 1118, 1126, 113, 16, 439, 347, 81, 404, 91, 507, 927, 343, 973, 120, 123, 759, 413, 76, 833, 315, 409, 322, 72, 1119, 1129, 700, 1266, 329, 912, 1114, 1237, 348, 239, 1050, 772, 109, 379, 574, 417, 531, 931, 357, 652, 1071, 1090, 1125, 210, 1060, 692, 256, 719, 943, 710, 581, 438, 342, 696, 296, 1178, 354, 345, 905, 566, 981, 125, 473, 952, 469, 1092, 84, 386, 337, 389, 724, 154, 632, 669, 508, 498, 254, 309, 362, 344, 577, 754, 1251, 637, 617, 1057, 880, 1107, 1061, 1171, 540, 57, 131, 643, 485, 806, 736, 1156, 430, 134, 1230, 639, 1241, 19, 408, 614, 807, 966, 906, 290, 1130, 277, 247, 180, 1235, 68, 548, 1131, 103, 1224, 743, 1167, 642, 1139, 1053, 1160, 655, 467, 698, 831, 779, 1063, 78, 584, 226, 693, 899, 129, 175, 135, 680, 785, 611, 933, 449, 872, 168, 1128, 1153, 328, 1270, 236, 1141, 1164, 75, 285, 546, 764, 468, 306, 303, 220, 260, 1267, 367, 712, 751, 390, 547, 56, 901, 504, 841, 114, 363, 518, 636, 159, 142, 761, 0, 1049, 563, 419, 156, 1216, 804, 1249, 519, 258, 433, 1127, 455, 883, 1115, 44, 1222, 373, 706, 311, 279, 703, 845, 463, 586, 608, 118, 1262, 350, 293, 172, 903, 460, 30, 849, 415, 934, 963, 371, 1043, 552, 957, 1079]
DEFAULT_FIXED_VAL_INDEX = [988, 112, 536, 986, 753, 1236, 888, 1154, 778, 189, 80, 936, 938, 1024, 1004, 453, 852, 1020, 28, 360, 1271, 1028, 1076, 1184, 213, 199, 208, 866, 1026, 998, 121, 238, 122, 741, 923, 1022, 664, 822, 1031, 71, 935, 160, 787, 7, 1083, 887, 526, 399, 53, 742, 929, 13, 612, 875, 1069, 697, 837, 176, 95, 786, 896, 191, 241, 217, 820, 885, 886, 6, 1032, 716, 1018, 1186, 598, 452, 163, 909, 1102, 1013, 17, 407, 882, 396, 824, 1214, 722, 924, 206, 74, 1001, 39, 179, 198, 1030, 993, 400, 1234, 398, 29, 1023, 676, 926, 982, 984, 521, 195, 523, 893, 675, 890, 707, 999, 691, 1263, 656, 465, 1029, 663, 1010, 1233, 193, 990, 748, 747, 1257, 731, 805, 671, 242, 451, 8, 721, 111, 38, 894, 895, 1005, 791, 904, 1196, 46, 524, 1172, 897, 970, 14, 1065, 454, 481, 1000, 907, 1199, 898, 879, 825, 1011, 794, 2, 1008, 687, 717, 525, 869, 165, 464, 212, 218, 194, 800, 528, 1003, 797, 1025, 1007, 944, 1093, 744, 599, 224, 3, 720, 1021, 200, 233, 995, 1146, 26, 1006, 726, 606, 204, 466, 244, 1019, 677, 937, 994, 9, 1027, 851, 763, 1124, 406, 873, 771, 600, 1088, 246, 1192, 830, 662, 859, 520, 157, 758, 216, 836, 397, 870, 63, 1207, 58, 425, 776, 939, 661, 395, 695, 650, 749, 1002, 119, 1209, 989, 605, 1017, 1015, 1188, 823, 185, 1238, 738, 527, 1084, 733, 50]
DEFAULT_FIXED_TEST_INDEX = [4, 5, 10, 12, 15, 18, 20, 21, 27, 31, 32, 34, 35, 36, 40, 41, 45, 52, 61, 64, 65, 66, 67, 69, 73, 77, 86, 104, 108, 115, 116, 117, 126, 132, 133, 136, 141, 143, 158, 162, 164, 167, 174, 177, 183, 184, 186, 187, 188, 190, 196, 197, 201, 203, 205, 207, 209, 214, 222, 223, 225, 228, 229, 230, 231, 232, 235, 237, 243, 245, 248, 252, 264, 265, 267, 268, 272, 276, 280, 281, 283, 288, 292, 294, 302, 305, 307, 308, 312, 314, 318, 323, 325, 327, 333, 334, 335, 340, 341, 346, 353, 365, 366, 369, 370, 372, 380, 394, 440, 456, 480, 514, 530, 532, 534, 535, 539, 543, 545, 549, 550, 555, 557, 558, 564, 565, 569, 573, 591, 595, 596, 597, 601, 602, 603, 604, 613, 621, 623, 626, 660, 662, 666, 670, 673, 676, 681, 688, 689, 701, 704, 711, 717, 723, 729, 730, 739, 740, 745, 748, 755, 756, 760, 767, 773, 788, 793, 795, 798, 809, 811, 815, 823, 824, 825, 827, 837, 838, 839, 842, 847, 853, 855, 857, 862, 865, 868, 871, 873, 910, 911, 914, 916, 918, 920, 945, 949, 950, 959, 960, 961, 962, 975, 976, 977, 978, 989, 991, 1009, 1017, 1044, 1062, 1068, 1074, 1077, 1080, 1081, 1082, 1085, 1086, 1087, 1089, 1091, 1095, 1098, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1112, 1123, 1138, 1143, 1147, 1150, 1151, 1152, 1155, 1161, 1162, 1163, 1173, 1174, 1175, 1176, 1177, 1179, 1180, 1181, 1183, 1191, 1194, 1200, 1213, 1215, 1232, 1239, 1250, 1255, 1256, 1265]


class STARSource:
    _SOURCE_CACHE: ClassVar[Dict[Tuple[Any, ...], "STARSource"]] = {}
    _RECORDS_CACHE: ClassVar[Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}

    def __init__(
        self,
        image_root: Path,
        roidb_file: Path,
        dict_file: Path,
        image_file: Path,
        image_ext: str = ".png",
    ):
        self.image_root = image_root
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.image_ext = image_ext

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = _load_star_info(self.dict_file)
        self.filenames, self.img_info = _load_star_image_filenames(
            self.image_root,
            self.image_file,
            image_ext=self.image_ext,
        )

        with h5py.File(self.roidb_file, "r") as roi_h5:
            self.num_images = len(roi_h5["split"][:])
            self.all_labels = roi_h5["labels"][:, 0].astype(np.int64)
            self.all_attributes = roi_h5["attributes"][:]
            self.all_boxes = roi_h5[f"boxes_{BOX_SCALE}"][:].astype(np.float32)
            self.all_boxes[:, 2:] = np.maximum(self.all_boxes[:, 2:], 1)
            self.all_boxes[:, :2] = self.all_boxes[:, :2] - self.all_boxes[:, 2:] / 2.0
            self.all_boxes[:, 2:] = self.all_boxes[:, :2] + self.all_boxes[:, 2:]
            seg_key = f"segmentation_{BOX_SCALE}"
            self.all_segments = roi_h5[seg_key][:] if seg_key in roi_h5 else None
            self.im_to_first_box = roi_h5["img_to_first_box"][:]
            self.im_to_last_box = roi_h5["img_to_last_box"][:]
            self.im_to_first_rel = roi_h5["img_to_first_rel"][:]
            self.im_to_last_rel = roi_h5["img_to_last_rel"][:]
            self.rel_pairs = roi_h5["relationships"][:]
            self.rel_predicates = roi_h5["predicates"][:, 0]

    @classmethod
    def from_paths(
        cls,
        image_root: Path,
        roidb_file: Path,
        dict_file: Path,
        image_file: Path,
        image_ext: str = ".png",
    ) -> "STARSource":
        cache_key = (
            str(image_root.resolve()),
            str(roidb_file.resolve()),
            str(dict_file.resolve()),
            str(image_file.resolve()),
            image_ext,
        )
        source = cls._SOURCE_CACHE.get(cache_key)
        if source is None:
            source = cls(image_root, roidb_file, dict_file, image_file, image_ext=image_ext)
            cls._SOURCE_CACHE[cache_key] = source
        return source

    @classmethod
    def clear_cache(cls) -> None:
        cls._SOURCE_CACHE.clear()
        cls._RECORDS_CACHE.clear()

    def get_split_records(
        self,
        split: str,
        box_mode: str = "hbb",
        split_mode: str = "fixed",
        split_ratios: Sequence[int] = (6, 2, 2),
        random_seed: int = 42,
        filter_empty_relations: bool = True,
        filter_non_overlap: bool = False,
        num_im: int = -1,
        box_coord_scale: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        cache_key = (
            str(self.roidb_file.resolve()),
            str(self.image_root.resolve()),
            str(self.image_file.resolve()),
            self.image_ext,
            split,
            box_mode,
            split_mode,
            tuple(int(v) for v in split_ratios),
            random_seed,
            filter_empty_relations,
            filter_non_overlap,
            num_im,
            None if box_coord_scale is None else float(box_coord_scale),
        )
        cached = self._RECORDS_CACHE.get(cache_key)
        if cached is None:
            cached = self._build_split_records(
                split=split,
                box_mode=box_mode,
                split_mode=split_mode,
                split_ratios=split_ratios,
                random_seed=random_seed,
                filter_empty_relations=filter_empty_relations,
                filter_non_overlap=filter_non_overlap,
                num_im=num_im,
                box_coord_scale=box_coord_scale,
            )
            self._RECORDS_CACHE[cache_key] = cached
        return cached

    def _build_split_records(
        self,
        split: str,
        box_mode: str,
        split_mode: str,
        split_ratios: Sequence[int],
        random_seed: int,
        filter_empty_relations: bool,
        filter_non_overlap: bool,
        num_im: int,
        box_coord_scale: Optional[float],
    ) -> List[Dict[str, Any]]:
        selected_indices = _select_split_indices(
            split=split,
            num_images=self.num_images,
            split_mode=split_mode,
            split_ratios=split_ratios,
            random_seed=random_seed,
        )

        records = []
        for image_index in selected_indices:
            first_box = int(self.im_to_first_box[image_index])
            last_box = int(self.im_to_last_box[image_index])
            if first_box < 0 or last_box < first_box:
                continue

            boxes_i = self.all_boxes[first_box : last_box + 1].copy()
            if box_mode == "obb":
                if self.all_segments is None:
                    raise KeyError(f"`segmentation_{BOX_SCALE}` not found in {self.roidb_file}")
                segment_i = self.all_segments[first_box : last_box + 1]
                boxes_i = _segments_to_obb(segment_i)

            classes_i = self.all_labels[first_box : last_box + 1].astype(np.int64)
            attributes_i = self.all_attributes[first_box : last_box + 1].copy()

            first_rel = int(self.im_to_first_rel[image_index])
            last_rel = int(self.im_to_last_rel[image_index])
            if first_rel >= 0 and last_rel >= first_rel:
                predicates_i = self.rel_predicates[first_rel : last_rel + 1].astype(np.int64)
                obj_idx = self.rel_pairs[first_rel : last_rel + 1] - first_box
                relations_i = np.column_stack((obj_idx, predicates_i)).astype(np.int64)
            else:
                relations_i = np.zeros((0, 3), dtype=np.int64)

            if filter_empty_relations and len(relations_i) == 0:
                continue
            if filter_non_overlap and len(relations_i) > 0:
                overlap_mask = _filter_non_overlapping_relations(boxes_i, relations_i, box_mode)
                if not overlap_mask.any():
                    continue
                relations_i = relations_i[overlap_mask]

            if image_index >= len(self.filenames):
                continue
            filename = self.filenames[image_index]
            if filename is None:
                continue

            img_meta = self.img_info[image_index]
            boxes_i = _rescale_boxes_to_image(
                boxes_i,
                img_meta,
                box_mode,
                scale=BOX_SCALE,
                target_coord_scale=box_coord_scale,
            )
            records.append(
                {
                    "image_index": image_index,
                    "file_name": Path(filename).name,
                    "width": int(img_meta["width"]),
                    "height": int(img_meta["height"]),
                    "boxes": boxes_i,
                    "labels": classes_i,
                    "attributes": attributes_i,
                    "relations": relations_i,
                }
            )

            if num_im > -1 and len(records) >= num_im:
                break

        return records

    def load_image(self, file_name: str):
        # STAR contains trusted, very large remote-sensing images.  Some test
        # images exceed twice Pillow's generic decompression-bomb threshold.
        return load_image_tensor(
            self.image_root / file_name,
            allow_large_images=True,
        )


def _load_star_info(dict_file: Path) -> Tuple[List[str], List[str], List[str]]:
    info = json.loads(dict_file.read_text(encoding="utf-8"))
    label_to_idx = dict(info["label_to_idx"])
    predicate_to_idx = dict(info["predicate_to_idx"])
    attribute_to_idx = dict(info.get("attribute_to_idx", {}))
    label_to_idx.setdefault("__background__", 0)
    predicate_to_idx.setdefault("__background__", 0)
    attribute_to_idx.setdefault("__background__", 0)

    classes = sorted(label_to_idx, key=lambda key: label_to_idx[key])
    predicates = sorted(predicate_to_idx, key=lambda key: predicate_to_idx[key])
    attributes = sorted(attribute_to_idx, key=lambda key: attribute_to_idx[key])
    return classes, predicates, attributes


def _load_star_image_filenames(
    image_root: Path,
    image_file: Path,
    image_ext: str = ".png",
) -> Tuple[List[Optional[str]], List[Dict[str, Any]]]:
    image_data = json.loads(image_file.read_text(encoding="utf-8"))
    filenames: List[Optional[str]] = []
    img_info: List[Dict[str, Any]] = []
    for item in image_data:
        basename = f"{int(item['image_id']):04d}{image_ext}"
        filename = image_root / basename
        filenames.append(str(filename) if filename.exists() else None)
        img_info.append(item)
    return filenames, img_info


def _select_split_indices(
    split: str,
    num_images: int,
    split_mode: str,
    split_ratios: Sequence[int],
    random_seed: int,
) -> List[int]:
    if split_mode == "random":
        train_idx, val_idx, test_idx = _random_split(num_images, ratios=split_ratios, seed=random_seed)
    elif split_mode == "fixed":
        train_idx, val_idx, test_idx = _load_fixed_split_indices()
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    mapping = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    return [idx for idx in mapping[split] if 0 <= idx < num_images]


def _random_split(
    num_samples: int,
    ratios: Sequence[int] = (6, 2, 2),
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)

    total = sum(ratios)
    n_train = int(num_samples * ratios[0] / total)
    n_val = int(num_samples * ratios[1] / total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _load_fixed_split_indices() -> Tuple[List[int], List[int], List[int]]:
    return (
        list(DEFAULT_FIXED_TRAIN_INDEX),
        list(DEFAULT_FIXED_VAL_INDEX),
        list(DEFAULT_FIXED_TEST_INDEX),
    )


def _rescale_boxes_to_image(
    boxes: np.ndarray,
    img_meta: Dict[str, Any],
    box_mode: str,
    scale: int = BOX_SCALE,
    target_coord_scale: Optional[float] = None,
) -> np.ndarray:
    width = max(float(img_meta["width"]), 1.0)
    height = max(float(img_meta["height"]), 1.0)
    target_scale = float(target_coord_scale) if target_coord_scale is not None else max(width, height)

    boxes = boxes.astype(np.float32).copy()
    if box_mode == "hbb":
        return boxes / float(scale) * target_scale

    scale_factor = target_scale / float(scale)
    boxes[:, :4] *= scale_factor
    return boxes


def _segments_to_obb(segments: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for STARDataset with `box_mode='obb'`.") from exc

    obb_boxes = []
    for segment in segments:
        points = np.asarray(segment[0], dtype=np.float32).reshape(4, 2)
        (cx, cy), (w, h), angle = cv2.minAreaRect(points)
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        # Match mmrotate.core.bbox.transforms.poly2obb_np_le90.
        # OpenCV returns degrees. Keep degrees here because STARDataset later
        # converts to radians when MODEL.OBB_ANGLE_UNIT == "radian".
        angle = float(angle)
        if w < h:
            w, h = h, w
            angle += 90.0
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        obb_boxes.append([float(cx), float(cy), w, h, angle])
    return np.asarray(obb_boxes, dtype=np.float32)


def _filter_non_overlapping_relations(
    boxes: np.ndarray,
    relations: np.ndarray,
    box_mode: str,
) -> np.ndarray:
    if len(relations) == 0:
        return np.zeros((0,), dtype=bool)

    hbb = boxes if box_mode == "hbb" else _obb_to_hbb_numpy(boxes)
    overlaps = _bbox_overlap_matrix(hbb, hbb) > 0
    keep = []
    for subj, obj, _ in relations.tolist():
        keep.append(bool(overlaps[int(subj), int(obj)]))
    return np.asarray(keep, dtype=bool)


def _obb_to_hbb_numpy(boxes: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for STARDataset with `box_mode='obb'`.") from exc

    hbbs = []
    for cx, cy, w, h, angle in boxes.tolist():
        points = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle)))
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        hbbs.append([x1, y1, x2, y2])
    return np.asarray(hbbs, dtype=np.float32)


def _bbox_overlap_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    return wh[:, :, 0] * wh[:, :, 1]
