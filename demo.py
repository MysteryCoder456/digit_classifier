import pygame
from keras.models import load_model

pygame.init()

# Load the classifier model
# model = load_model("model.keras", compile=False)

# Intialize pygame stuff

WIN_SIZE = (500, 600)
win = pygame.display.set_mode(WIN_SIZE)
pygame.display.set_caption("Digits Neural Network Demo")

font = pygame.font.Font(None, 40)
clock = pygame.time.Clock()
fps = 60


def main():
    while True:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        win.fill(70)

        text = font.render("Prediction: TODO", False, "white")
        text_rect = text.get_rect(center=(WIN_SIZE[0] / 2, 25))
        win.blit(text, text_rect)

        pygame.display.flip()


if __name__ == "__main__":
    main()
