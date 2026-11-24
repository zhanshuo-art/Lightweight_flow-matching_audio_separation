# Week5 成功/失败案例分析

本文件从现有 20 个试听样例中选择案例，不重新采样、不重新训练。选择依据为 Week4 Mamba 与 Week5 Mamba FT 的 target SI-SDR 差异，以及 Spec L1 与 SI-SDR 是否同向。

| Type | Sample | Mixture | Target | Residual | Week4 SI-SDR | Week5 SI-SDR | Week5 Residual SI-SDR | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| success | sample_10 / test_000006_t0 | rain + dog | rain | dog | 2.51 | 3.51 | 9.40 | Week5 target SI-SDR 比 Week4 提升 1.00 dB。该样例中 target=rain，residual=dog，模型能较好利用 target label 抑制非目标声源。 |
| success | sample_04 / test_000003_t0 | keyboard_typing + vacuum_cleaner | keyboard_typing | vacuum_cleaner | -0.22 | 0.32 | 4.15 | Week5 target SI-SDR 比 Week4 提升 0.54 dB。该样例中 target=keyboard_typing，residual=vacuum_cleaner，模型能较好利用 target label 抑制非目标声源。 |
| success | sample_13 / test_000007_t1 | keyboard_typing + door_wood_knock | door_wood_knock | keyboard_typing | 1.39 | 2.15 | 1.68 | Week5 target SI-SDR 比 Week4 提升 0.75 dB。该样例中 target=door_wood_knock，residual=keyboard_typing，模型能较好利用 target label 抑制非目标声源。 |
| failure | sample_00 / test_000001_t0 | clock_tick + keyboard_typing | clock_tick | keyboard_typing | -4.40 | -8.01 | 8.38 | Week5 target SI-SDR 相对 Week4 下降 3.61 dB。可能原因是：瞬态声音很短，预测幅度和 mixture phase 的误差会被放大。 |
| failure | sample_06 / test_000004_t0 | glass_breaking + car_horn | glass_breaking | car_horn | -1.61 | -4.27 | 12.56 | Week5 target SI-SDR 相对 Week4 下降 2.66 dB。可能原因是：瞬态声音很短，预测幅度和 mixture phase 的误差会被放大。 |
| failure | sample_02 / test_000002_t0 | rain + siren | rain | siren | 4.40 | 2.51 | 4.99 | Week5 target SI-SDR 相对 Week4 下降 1.89 dB。可能原因是：目标声源和干扰声源容易在时间或频谱上重叠。 |
| borderline | sample_01 / test_000001_t1 | clock_tick + keyboard_typing | keyboard_typing | clock_tick | -6.34 | 0.80 | -24.45 | 该样例的 Spec L1 变化为 +0.0012，但 SI-SDR 变化为 +7.14 dB。这说明单一频谱 L1 与最终音频质量指标并不总是同向。 |
| borderline | sample_07 / test_000004_t1 | glass_breaking + car_horn | car_horn | glass_breaking | 2.98 | 5.23 | -9.29 | 该样例的 Spec L1 变化为 +0.0065，但 SI-SDR 变化为 +2.24 dB。这说明单一频谱 L1 与最终音频质量指标并不总是同向。 |

## 音频路径

### success - sample_10 (test_000006_t0)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_10/week5_prediction_residual.wav`

### success - sample_04 (test_000003_t0)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_04/week5_prediction_residual.wav`

### success - sample_13 (test_000007_t1)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_13/week5_prediction_residual.wav`

### failure - sample_00 (test_000001_t0)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_00/week5_prediction_residual.wav`

### failure - sample_06 (test_000004_t0)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_06/week5_prediction_residual.wav`

### failure - sample_02 (test_000002_t0)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_02/week5_prediction_residual.wav`

### borderline - sample_01 (test_000001_t1)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_01/week5_prediction_residual.wav`

### borderline - sample_07 (test_000004_t1)

- mixture: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/mixture.wav`
- target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/target.wav`
- residual target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/residual_target.wav`
- Week4 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/week4_prediction_target.wav`
- Week5 target: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/week5_prediction_target.wav`
- Week5 residual: `D:/evado_onedrive/OneDrive/Desktop/通识公选课/深度学习入门/深度学习大作业/week5/analysis/week5_listening_demo/assets/sample_07/week5_prediction_residual.wav`
