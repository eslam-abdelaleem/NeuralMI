# Potential Issues and Areas for Improvement

This document lists potential issues, design concerns, and areas for future improvement identified during the comprehensive review and testing of the `neural_mi` library.

## 1. `ConcatCriticCNN` Docstring Mismatch (Fixed)

*   **Issue:** The docstring for `ConcatCriticCNN` in `neural_mi/models/critics.py` incorrectly stated that the model "concatenates the feature maps" from the two CNN towers.
*   **Actual Behavior:** The implementation concatenates the final *embedding vectors* produced by the CNNs, not the intermediate feature maps.
*   **Status:** **Fixed.** The docstring has been updated to accurately reflect the model's behavior.

## 2. `CNN1D` Dynamic Layer Creation

*   **Issue:** The `CNN1D` embedding model uses "lazy initialization" to create its final fully-connected layers on the first forward pass. While this allows for handling variable-length inputs, it has several drawbacks:
    *   **Not Thread-Safe:** If the model is used in a multi-threaded or parallel processing environment, it could lead to race conditions where multiple threads try to initialize the layers simultaneously.
    *   **Breaks State Dict:** The model's state dictionary changes after the first forward pass, which can complicate saving and loading model checkpoints. A model's architecture should ideally be fixed after instantiation.
    *   **Reduced Clarity:** This behavior is non-standard and can make the model harder to understand, debug, and integrate with other frameworks.
*   **Recommendation:** Consider refactoring `CNN1D` to have a fixed architecture. This could be achieved by requiring the input sequence length to be specified during initialization or by using global pooling layers (e.g., `nn.AdaptiveAvgPool1d`) to produce a fixed-size output from the convolutional base, regardless of the input length.

## 3. `VarMLP` Stateful Forward Pass

*   **Issue:** The `forward` method of the `VarMLP` model has a side effect: it updates the `self.kl_loss` attribute when the model is in `training` mode.
*   **Problem:** This is a stateful design that is not transparent and can lead to bugs. For example, if a user forgets to manually retrieve and zero out the `kl_loss` after each training step, the loss could accumulate incorrectly.
*   **Recommendation:** A cleaner, functional approach would be to have the `forward` method return both the embedding and the KL loss. For example: `return z, kl_loss`. This makes the model's behavior explicit and stateless.

## 4. `js_fgan_lower_bound` Instability

*   **Issue:** The `js_fgan_lower_bound` estimator proved to be unstable during testing. Unlike other estimators, it produced a negative mutual information estimate for highly correlated data when using a randomly initialized critic.
*   **Problem:** This suggests that the `js_fgan_lower_bound` may be more sensitive to the quality of the critic model than other estimators. It might require a well-trained (or pre-trained) critic to produce reliable and meaningful positive MI estimates.
*   **Recommendation:** The documentation for this estimator should include a warning about its potential instability and a recommendation to use it with a well-trained critic. The current test suite isolates this estimator to prevent spurious failures, but a more advanced test involving a pre-trained critic could be developed to validate it more thoroughly.