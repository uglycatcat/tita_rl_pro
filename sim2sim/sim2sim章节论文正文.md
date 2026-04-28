此md文档是我毕业设计论文正文的内容
此处先做一个大概的介绍。我的毕业设计包括了当我在isaac gym环境下训练得到模型之后。sim2sim的相关工作。
在这部分工作中我的主要工作内容包括
搭建sim2sim的代码管线，创建mjcf文件，让机器人在mujoco环境下正确加载onnx模型并且正常进行运动。、
代码集中在sim2sim文件夹中

之后主要做了三个实验

首先是分别在isaac gym和mujoco下做持续前进任务，保持双方环境尽可能一致的情况下收集两端运动和推理相关信息参数。用于验证sim2sim的效果和必要性。
一套只看 r0/r1、偏论文展示的图，重点突出 MuJoCo 优势。

生成位置：

sim2sim/results/mujoco_vs_phyX/figure_paper/paper_r01_core_bar.png
sim2sim/results/mujoco_vs_phyX/figure_paper/paper_r01_mujoco_advantage_percent.png
sim2sim/results/mujoco_vs_phyX/figure_paper/paper_r01_action_delta_line.png
sim2sim/results/mujoco_vs_phyX/figure_paper/paper_r01_tracking_error_lines.png
新增脚本：

sim2sim/results/mujoco_vs_phyX/plot_paper_mujoco_advantage.py
这 4 张图分别是：

核心指标柱状图（vx_mae/yaw_mae/roll_rms/pitch_rms/action_delta*）
MuJoCo 相对 PhysX 的提升百分比柱状图
动作平滑度折线图（action_delta_l2 rolling mean）
速度与偏航跟踪误差折线图（绝对误差 rolling mean）
如果你要，我下一步可以再给你补一版“论文图注模板”（中英双语，每张图 2-3 句可直接贴到论文里）。

这四张图可以这样解读（都基于 r0/r1）：

核心指标柱状图
对比 MuJoCo 和 PhysX 在关键指标上的均值（带重复间波动）。

vx_mae：前向速度跟踪误差，越低越好
yaw_mae：偏航角速度跟踪误差，越低越好
roll_rms / pitch_rms：姿态滚转/俯仰抖动，越低越稳
action_delta_l2_mean / action_delta_rate_mean：动作步间变化幅度与变化速率，越低越平滑
这张图是“总览图”，看整体谁更好。
MuJoCo 相对 PhysX 的提升百分比柱状图
把“差多少”转成百分比：

$\mathrm{improvement}=\frac{\mathrm{PhysX}-\mathrm{MuJoCo}}{\mathrm{PhysX}}\times100\%$，
正值：MuJoCo 更优（误差更小）
负值：MuJoCo 更差
这张图适合论文里一句话量化结论，比如“在 yaw 跟踪误差上降低 xx%”。
动作平滑度折线图（action_delta_l2 rolling mean）
展示随时间的动作跳变趋势（滚动平均后更平滑、可读性更高）。

曲线越低：控制输出越连续、越不抖
阴影带：r0/r1 两次重复的离散程度
这张图强调“MuJoCo 的控制信号更稳定”。
速度与偏航跟踪误差折线图（绝对误差 rolling mean）
两个子图：

左：$\left|v_x-v_{x,\mathrm{cmd}}\right|$，
右：$\left|\omega_{\mathrm{yaw}}-\omega_{\mathrm{yaw},\mathrm{cmd}}\right|$
曲线越低表示跟踪越准，阴影带表示重复间波动。
这张图强调“时间维度上谁在持续更好地跟踪命令”。
如果你愿意，我可以再给你每张图对应的论文图注模板（中文+英文），你可以直接贴到论文里。












其次是单独在mujoco环境下让机器人在不同的地形做相同任务，并观察他们的运动情况来证明训练结果的泛化性
已在 sim2sim/results/terrain_contrast/plot_terrain_contrast.py 里补充了 3 张更适合论文表达“地形泛化能力”的图，并已实际生成到 sim2sim/results/terrain_contrast/figure/。

新增图像：

terrain_generalization_retention.png

含义：以 flat 为基准（=1.0），比较 slope/stairs 在跟踪、姿态、存活、平滑性上的“能力保持率”。
论文价值：可以直接说明“复杂地形有退化，但仍保留一定能力”，并且定量展示退化幅度。
terrain_repeat_variability.png

含义：对 vx_mae / survival_time / action_delta_rate 做按地形的箱线+散点（每个 repeat 一点）。
论文价值：体现泛化不只看均值，还看稳定性与方差，能回答“是否偶然好结果”。
terrain_vx_error_cdf.png

含义：三种地形上 |base_lin_vx - cmd_vx| 的累计分布曲线（CDF）。
论文价值：从分布层面展示 flat 最集中、slope/stairs 右移，直观看出泛化退化趋势。














最后是进行鲁棒性测试，通过动力学鲁棒性测试观察mujoco环境下的机器人运动情况来证明训练结果的鲁棒性
要的目标，在 sim2sim/results/robustness/ 补了更能支撑“训练结果具有一定鲁棒性”的论文图，并已生成完成。

这次我修改了 sim2sim/results/robustness/plot_robustness.py，新增 3 张图：

sim2sim/results/robustness/figure/robustness_retention_ratio.png

展示 Dynamics Rand / Baseline 的能力保持率（VX Track、Yaw Track、Survival、Smoothness）。
一眼可见哪些地形在随机化后还能保持接近 1.0，哪些出现明显退化。
sim2sim/results/robustness/figure/robustness_vx_error_cdf.png

按 flat/slope/stairs 分子图，对比 baseline 与随机化下的速度跟踪误差 CDF。
比均值更适合论文，能体现“尾部风险”和整体分布右移/左移。
sim2sim/results/robustness/figure/robustness_survival_fall.png

同时给出存活时长和跌倒率（均值+方差）。
直接回答鲁棒性中最关心的稳定性问题。
另外，原有图也会一起继续输出（core/degradation/heatmap/paired scatter/timeseries 等），现在脚本一跑就是完整鲁棒性图包。
我需要你把我的整个工作撰写成论文的一个章节，包括意义，目标，实践过程，实验结果及分析，展望等等。你是在写论文的正文，少提代码，多表达逻辑和学术内容。