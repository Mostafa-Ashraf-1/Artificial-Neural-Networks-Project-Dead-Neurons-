# import supervision as sv
# import numpy as np
# import os
# class DataSplit:

#     def __init__(self, image_path, annotation_path, new_file_name):
#         self.image_path = image_path
#         self.annotation_path = annotation_path
#         self.new_file_name = new_file_name

#     def split_data(self):
#         # 1. Load the dataset
#         ds = sv.DetectionDataset.from_coco(
#             images_directory_path= self.image_path,
#             annotations_path= self.annotation_path,
#         )

#         # 2. Fix the coordinate type (prevents the TypeError you saw earlier)
#         for image_name in ds.annotations:
#             ds.annotations[image_name].xyxy = ds.annotations[image_name].xyxy.astype(float)

#         # 3. First Split: Separate Training (70%) from the rest (30%)
#         train_ds, remaining_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)

#         # 4. Second Split: Split the remaining 30% into Validation and Test (50/50 split of the 30%)
#         val_ds, test_ds = remaining_ds.split(split_ratio=0.5, random_state=42, shuffle=True)

#         print(f"Final Totals: Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

#         # 5. Export all three sets to disk
#         splits = {
#             "train": train_ds,
#             "valid": val_ds,
#             "test": test_ds
#         }

#         for name, dataset in splits.items():
#             output_path = f"{self.new_file_name}/{name}"
#             os.makedirs(f"{output_path}/images", exist_ok=True)
            
#             dataset.as_coco(
#                 images_directory_path=f"{output_path}/images",
#                 annotations_path=f"{output_path}/_annotations.coco.json"
#             )

#         print(f"Done! You now have train, valid, and test folders in '{self.new_file_name}/'.")

#         # print("Train classes:", train_ds.classes)
#         # print("Valid classes:", val_ds.classes)
#         # print("Test classes:", test_ds.classes)

import supervision as sv
import numpy as np
import os


class DataSplit:

    def __init__(self, image_path, annotation_path, new_file_name):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.new_file_name = new_file_name

    def _validate_boxes(self, dataset, name="dataset"):
        """
        Ensure all bounding boxes are valid:
        x2 > x1 and y2 > y1
        """
        broken = 0

        for image_name, ann in dataset.annotations.items():
            boxes = ann.xyxy

            if len(boxes) == 0:
                continue

            invalid = np.where(
                (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            )[0]

            if len(invalid) > 0:
                print(f"[WARNING] Invalid boxes found in {name} -> {image_name}")
                broken += len(invalid)

        if broken == 0:
            print(f"[OK] All bounding boxes in {name} are valid ✅")
        else:
            print(f"[ERROR] Found {broken} invalid boxes in {name} ❌")

    def split_data(self):
        # 1. Load dataset (DO NOT TOUCH COORDINATES)
        ds = sv.DetectionDataset.from_coco(
            images_directory_path=self.image_path,
            annotations_path=self.annotation_path,
        )

        print(f"Loaded dataset with {len(ds)} images")

        # 2. Validate BEFORE splitting
        self._validate_boxes(ds, "original dataset")

        # 3. Split: 70% train / 30% remaining
        train_ds, remaining_ds = ds.split(
            split_ratio=0.7,
            random_state=42,
            shuffle=True
        )

        # 4. Split remaining into 15% val / 15% test
        val_ds, test_ds = remaining_ds.split(
            split_ratio=0.5,
            random_state=42,
            shuffle=True
        )

        print(f"Final Totals:")
        print(f"Train: {len(train_ds)}")
        print(f"Valid: {len(val_ds)}")
        print(f"Test : {len(test_ds)}")

        # 5. Validate AFTER splitting
        self._validate_boxes(train_ds, "train")
        self._validate_boxes(val_ds, "valid")
        self._validate_boxes(test_ds, "test")

        # 6. Export
        splits = {
            "train": train_ds,
            "valid": val_ds,
            "test": test_ds
        }

        for name, dataset in splits.items():
            output_path = os.path.join(self.new_file_name, name)
            images_path = os.path.join(output_path, "images")

            os.makedirs(images_path, exist_ok=True)

            dataset.as_coco(
                images_directory_path=images_path,
                annotations_path=os.path.join(output_path, "_annotations.coco.json")
            )

        print(f"\n✅ Done! Dataset saved to '{self.new_file_name}/'")