import pygame
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

pygame.init()

# Define constants
WIN_SIZE = (500, 550)
IMAGE_SIZE = (28, 28)
DRAWING_AREA_SIZE = (400, 400)
FPS = 100

# Load the classifier model
model = load_model("model.keras", compile=False)

# Intialize pygame stuff

win = pygame.display.set_mode(WIN_SIZE)
pygame.display.set_caption("Digits Neural Network Demo")

font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()
running = True

image = np.zeros(IMAGE_SIZE)
image_surface = pygame.Surface(DRAWING_AREA_SIZE)
image_surface.fill("black")

prediction = None
confidence = 0


while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()

            if keys[pygame.K_c]:
                # Clear drawing area
                image = np.zeros(IMAGE_SIZE)
                image_surface.fill("black")

            if keys[pygame.K_p]:
                # Plot current image data
                plt.imshow(image)
                plt.show()

        elif event.type == pygame.MOUSEBUTTONUP:
            pred = model.predict(np.array([image]))[0]  # type: ignore
            prediction = np.argmax(pred)
            confidence = pred[prediction]

    image_rect = image_surface.get_rect(
        center=(
            WIN_SIZE[0] / 2,
            WIN_SIZE[1] - DRAWING_AREA_SIZE[1] / 2 - 50,
        )
    )
    mouse_pressed = any(pygame.mouse.get_pressed())
    mouse_pos = pygame.mouse.get_pos()

    if mouse_pressed and image_rect.collidepoint(mouse_pos):
        pixel_size = (
            DRAWING_AREA_SIZE[0] / IMAGE_SIZE[0],
            DRAWING_AREA_SIZE[1] / IMAGE_SIZE[1],
        )

        relative_x = mouse_pos[0] - image_rect.x
        relative_y = mouse_pos[1] - image_rect.y
        x_idx = int(relative_x // pixel_size[0])
        y_idx = int(relative_y // pixel_size[1])

        image[y_idx, x_idx] = 255
        pygame.draw.rect(
            image_surface,
            "white",
            (
                x_idx * pixel_size[0],
                y_idx * pixel_size[1],
                # +1 to avoid weird lines in between
                pixel_size[0] + 1,
                pixel_size[1] + 1,
            ),
        )

    win.fill(70)

    # Render prediction
    text = font.render(
        f"Prediction: {prediction}  Confidence: {confidence*100:.2f}%",
        False,
        "white",
    )
    text_rect = text.get_rect(center=(WIN_SIZE[0] / 2, 50))
    win.blit(text, text_rect)

    # Render drawing area
    win.blit(image_surface, image_rect)

    pygame.display.flip()
