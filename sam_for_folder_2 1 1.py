import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask_on_image(image, mask):
    alpha = 0.5  # Adjust the transparency level
    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)  # Set overlay color (green in this case)
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return output


sam = sam_model_registry["vit_h"](checkpoint="/home/zestiot/Downloads/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

click_x, click_y = -1, -1


def on_mouse_click(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONUP:
        click_x, click_y = x, y
        print(f"Clicked at: ({click_x}, {click_y})")


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


folder_path = "/home/zestiot/Desktop/Zestiot/PROJECTS/P_G/Tao"  # Replace with your folder path
output_folder = "/home/zestiot/Desktop/Zestiot/PROJECTS/P_G/sam_output"  # Replace with your desired output folder path

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image before entering the loop
    predictor.set_image(image)

    response = 'yes'  # Start the loop
    all_points = []
    all_labels = []

    while response.lower() == 'yes':
        cv2.imshow('Base Image', image)
        cv2.setMouseCallback('Base Image', on_mouse_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        input_point = np.array([[click_x, click_y]])
        input_label = np.array([1])
        all_points.append(input_point)
        all_labels.append(input_label)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for points, labels in zip(all_points, all_labels):
            show_points(points, labels, plt.gca())
        plt.axis('on')
        plt.show()

        masks, scores, logits = predictor.predict(
            point_coords=np.vstack(all_points),
            point_labels=np.concatenate(all_labels),
            multimask_output=True,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            ref = np.zeros(mask.shape, np.uint8)
            ref[mask] = 255
            print(ref)
            for points, labels in zip(all_points, all_labels):
                show_points(points, labels, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

        response = input('Add one more point? (yes/no): ')

    if all_points:  # Check if any points were added
        chosen_mask = int(input('Which mask did you like? (Enter the number): ')) - 1
        chosen_mask_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_chosen_mask.jpg")

        # Overlay the chosen mask on the original image
        masked_image = overlay_mask_on_image(image, masks[chosen_mask])

        cv2.imwrite(chosen_mask_path, masked_image)
        print(f"Mask overlay saved at: {chosen_mask_path}")
    else:
        print("No masks saved for this image.")

print("Segmentation process completed.")
