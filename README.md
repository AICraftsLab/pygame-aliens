This project converts the built-in Pygame's [Aliens](https://www.pygame.org/docs/ref/examples.html#pygame.examples.aliens.main) game to a gymnasium enviromnent which can be used to train AI where gymnasium environments are supported. Check *test_env.py* on how to make the env.

AI model is trained using Stable-Baselines3, check *sb3_trainer.py* to play the game. After 15 million steps, the AI mostly plays really well. The 15-million-steps trained model file is inside *run1* folder along side 5-million- and 10-million-steps checkpoints. Run *sb3_inference.py* to see the model playing.

![Image](https://github.com/user-attachments/assets/978b0ff7-64ec-4e17-bfdc-92d1f8301ab6)

The env is truncated when the score reaches 100. Clicking the game screen will toggle lines showing some of the observations taken by the model.
