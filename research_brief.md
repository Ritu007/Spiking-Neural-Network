# Research Brief: Spiking Neural Networks (SNNs)

## State of the Art (2026)
- **Neuron Models**: Evaluation of Adaptive LIF (ALIF) vs. standard LIF for long-term dependency tasks.
- **Learning Algorithms**: Analysis of Backpropagation Through Time (BPTT) with surrogate gradients vs. local learning rules like e-prop.

## Application Shortlist
1. **Event-Based Object Detection & Tracking**: Utilizing dynamic vision sensors (DVS) to process asynchronous visual events. Highly implementable using `SpikingJelly`'s neuromorphic dataset loaders (e.g., DVS128 Gesture, N-Caltech101) for real-time inference with minimal latency.
2. **Ultra-Low-Power Keyword Spotting (KWS)**: Processing audio streams converted into spike trains for "always-on" edge devices. Models can be easily trained via `snnTorch` using surrogate gradients on datasets like Google Speech Commands.
3. **Real-time IoT Anomaly Detection**: Processing continuous time-series sensor data from industrial or edge devices. SNNs provide an energy-efficient way to detect temporal anomalies in sparse data, easily deployable as a standalone system.
4. **Brain-Machine Interface (BMI) Signal Decoding**: Decoding physiological signals like EEG or intracortical neural spikes for prosthetic control. The temporal dynamics of SNNs natively match the biological data, making frameworks like `snnTorch` ideal for BMI research.
5. **Spiking Autonomous Navigation**: Implementing obstacle avoidance and basic pathfinding for small robots/drones using event-camera inputs mapped to motor control commands, offering a high-efficiency alternative to traditional CNNs.

## Comparative Metrics
- **Energy Efficiency**: Spikes per inference vs. Floating Point Operations (FLOPs).
- **Latency**: Time-to-first-spike (TTFS) vs. traditional frame-based latency.
