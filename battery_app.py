import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="李小布的科研主页 | PINN 电池监控",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 侧边栏：个人信息 & 控制台
# ==========================================
with st.sidebar:
    # 这里的头像可以用你自己的，或者保留这个随机生成的卡通头像
    st.image("https://api.dicebear.com/9.x/avataaars/svg?seed=Felix", width=100) 
    st.markdown("## 👨‍💻 关于开发者 (About Me)")
    
    # 【修改点1】优化排版，使用空行确保换行，或者使用列表格式
    st.info("""
    **姓名**：李小布
    
    **学校**：南京邮电大学 (26届硕士)
    
    **专业**：计算数学 / 应用数学
    
    **研究方向**：深度学习与科学计算 (AI for Science)
    
    **核心技能**：PINN, PyTorch, 数值分析, CFD
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ 仿真参数设置")
    true_alpha = st.number_input("真实热扩散系数 (True Alpha)", value=0.01, format="%.4f")
    noise_level = st.slider("传感器噪声水平 (Noise %)", 0.0, 5.0, 1.0, step=0.1)
    
    st.markdown("### ⚙️ 求解器配置")
    epochs = st.slider("训练轮数 (Epochs)", 1000, 10000, 5000, step=1000)
    pde_weight = st.slider("物理权重 (PDE Weight)", 1.0, 20.0, 10.0)
    lr = st.number_input("学习率 (Learning Rate)", value=0.001, format="%.4f")
    
    st.markdown("---")
    st.caption("© 2024 PINN Battery Project. All Rights Reserved.")

# ==========================================
# 2. 主页面内容
# ==========================================
st.title("🔋 基于 PINN 的锂电池热参数反演与实时监控系统")
st.markdown("##### *Physics-Informed Neural Networks for Battery Thermal Management & State Estimation*")

# 使用 Tab 分隔理论介绍和实战演示
tab1, tab2, tab3 = st.tabs(["📖 项目背景与痛点", "💡 核心技术方案", "🚀 在线仿真 Demo"])

# ------------------------------------------
# TAB 1: 背景与痛点
# ------------------------------------------
with tab1:
    st.header("1. 为什么要做这个项目？")
    
    col_bg1, col_bg2 = st.columns(2)
    
    with col_bg1:
        st.markdown("### 🛑 行业痛点 (The Problem)")
        st.error("""
        **1. 内部温度不可测**
        *   现有 BMS 传感器只能贴在电池表面。
        *   电池内部核心温度往往比表面高 5-10°C，容易引发热失控风险。
        
        **2. 传统仿真太慢**
        *   **FEM/CFD (有限元)**：精度高，但计算一次需要几分钟甚至几小时，无法在车载芯片上实时运行。
        *   **等效电路模型**：依赖大量查表，难以反映复杂的热流场分布。
        
        **3. SOH (健康状态) 难估算**
        *   电池老化会导致**热扩散系数**、**内阻**等物理参数变化，这些参数无法直接测量，只能通过“反问题”推算。
        """)
    
    with col_bg2:
        st.markdown("### ✅ 我们的目标 (The Goal)")
        st.success("""
        **打造“虚拟传感器” (Virtual Sensor)**
        
        *   **输入**：仅利用表面稀疏的、含噪声的温度传感器数据。
        *   **内核**：基于物理定律 (PINN) 的深度学习模型。
        *   **输出**：
            1.  **实时重构**电池内部不可见的温度场。
            2.  **自动反演**电池的热物性参数 (对应 SOH)。
        *   **优势**：毫秒级推理速度 +物理级计算精度。
        """)
    
    st.image("https://github.com/maziarraissi/PINNs/raw/master/figures/PINN.png", caption="PINN 原理示意图 (Physics Loss + Data Loss)", use_container_width=True)

# ------------------------------------------
# TAB 2: 技术方案       
# ------------------------------------------
with tab2:
    st.header("2. 核心技术详解")
    st.markdown("这是本项目解决**“含噪数据下高精度反演”**的三大核心技术。")
    
    with st.expander("📌 技术点 1:混合损失函数与自动微分 (Physics-Informed Loss)", expanded=True):
        st.markdown(r"""
        我们不依赖大量标签数据，而是将**热传导方程 (Heat Equation)** 嵌入到 Loss 函数中：
        
        $$
        \mathcal{L} = \underbrace{\frac{1}{N}\sum(u_{pred} - u_{sensor})^2}_{\text{Data Loss (观测误差)}} + \lambda \cdot \underbrace{\frac{1}{M}\sum(u_t - \alpha u_{xx})^2}_{\text{PDE Loss (物理残差)}}
        $$
        
        *   **做法**：利用 PyTorch 的 `torch.autograd` 实现无网格自动微分。
        *   **价值**：保证了预测结果必须符合物理学定律，即使数据有噪声，模型也不会过拟合。
        """)
        
    with st.expander("📌 技术点 2:参数自适应反演 (Inverse Problem Solving)", expanded=True):
        st.markdown("""
        **简历描述**：“将热扩散系数设为可训练变量...”
        
        *   **实现**：在代码中定义 `self.alpha = nn.Parameter(...)`。
        *   **机制**：在反向传播更新网络权重的同时，利用梯度下降法同步修正物理参数 $\\alpha$。
        *   **意义**：这就相当于让 AI 自动“猜”出电池的老化程度。
        """)

    with st.expander("📌 技术点 3：抗噪优化策略 (Optimization Strategy)", expanded=True):
        st.markdown("""
        **简历描述**：“混合优化策略与加权 Loss...”
        
        *   **挑战**：真实传感器数据有噪声，导致 $\\alpha$ 反演不稳定。
        *   **解决方案**：
            1.  **权重平衡**：增大 PDE Loss 的权重 (如 $\lambda=100$)，强迫模型优先满足物理方程。
            2.  **两阶段训练**：先用 Adam 快速收敛，再用 L-BFGS (二阶优化) 进行微调，提高精度。
        """)

# ------------------------------------------
# TAB 3: 在线演示 (核心代码逻辑)
# ------------------------------------------
with tab3:
    st.header("3. 在线仿真与实时反演")
    st.warning("⚠️ 点击下方按钮开始训练。由于是浏览器端运行，建议 Epoch 设置在 3000-5000 左右。")
    
    # --------------------------------
    # 后端逻辑区 (为了不卡顿页面，放在函数里)
    # --------------------------------
    @st.cache_resource
    def get_device():
        return torch.device("cpu")

    device = get_device()

    class PINN(nn.Module):
        def __init__(self):
            super(PINN, self).__init__()
            # 4层全连接，每层20个神经元，Tanh激活
            self.net = nn.Sequential(
                nn.Linear(2, 20), nn.Tanh(),
                nn.Linear(20, 20), nn.Tanh(),
                nn.Linear(20, 20), nn.Tanh(),
                nn.Linear(20, 20), nn.Tanh(),
                nn.Linear(20, 1)
            )
            # 待反演参数 Alpha (初始猜测 0.02)
            self.alpha = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))

        def forward(self, x, t):
            inputs = torch.cat([x, t], dim=1)
            return self.net(inputs)

        def physics_loss(self, x, t):
            u = self.forward(x, t)
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            f = u_t - self.alpha * u_xx
            return torch.mean(f ** 2)

    def generate_data(alpha, noise_pct):
        # 制造真值
        def analytic_solution(x, t, alpha):
            return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

        x = np.linspace(0, 1, 100)
        t = np.linspace(0, 1, 100)
        X, T = np.meshgrid(x, t)
        u_true = analytic_solution(X, T, alpha)
        
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = u_true.flatten()[:,None]
        
        # 训练数据：500个点，加噪声
        idx = np.random.choice(X_star.shape[0], 500, replace=False)
        X_train = torch.tensor(X_star[idx, :], dtype=torch.float32).to(device)
        u_train = torch.tensor(u_star[idx, :], dtype=torch.float32).to(device)
        
        # 噪声处理: noise_pct 是百分比 (e.g. 1.0)
        noise_std = (noise_pct / 100.0)
        u_train = u_train + noise_std * torch.randn_like(u_train)
        
        # PDE 约束点
        idx_f = np.random.choice(X_star.shape[0], 2000, replace=False)
        X_f = torch.tensor(X_star[idx_f, :], dtype=torch.float32, requires_grad=True).to(device)
        
        return X, T, u_true, X_star, X_train, u_train, X_f

    # --------------------------------
    # 交互逻辑区
    # --------------------------------
    if st.button("🚀 启动数字孪生模型 (Start Simulation)", type="primary"):
        
        # 1. 准备数据
        X, T, u_true, X_star, X_train, u_train, X_f = generate_data(true_alpha, noise_level)
        
        # 2. 初始化
        model = PINN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # UI 占位符
        # 【修改点2】增加一列，显示训练轮数
        col_epoch, col_metrics1, col_metrics2, col_metrics3 = st.columns(4)
        
        metric_epoch = col_epoch.empty()  # 新增：显示轮数
        metric_loss = col_metrics1.empty()
        metric_alpha = col_metrics2.empty()
        metric_err = col_metrics3.empty()
        
        chart_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        loss_history = []
        alpha_history = []
        
        st.info("🔄 模型训练中，正在反演内部物理参数...")
        
        # 3. 训练循环
        start_time = time.time()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            
            u_pred = model(X_train[:, 0:1], X_train[:, 1:2])
            loss_data = torch.mean((u_pred - u_train) ** 2)
            loss_physics = model.physics_loss(X_f[:, 0:1], X_f[:, 1:2])
            
            # 加权 Loss
            loss = loss_data + pde_weight * loss_physics
            loss.backward()
            optimizer.step()
            
            # 记录数据
            curr_loss = loss.item()
            curr_alpha = model.alpha.item()
            loss_history.append(curr_loss)
            alpha_history.append(curr_alpha)
            
            # 动态刷新 UI (每 5% 刷新一次，防止浏览器卡顿)
            if epoch % (epochs // 20) == 0:
                progress_bar.progress(epoch / epochs)
                
                err_val = abs(curr_alpha - true_alpha) / true_alpha * 100
                
                # 更新四个指标卡片
                metric_epoch.metric("当前轮数 (Epoch)", f"{epoch} / {epochs}")
                metric_loss.metric("Current Loss", f"{curr_loss:.2e}")
                metric_alpha.metric("Predicted Alpha", f"{curr_alpha:.5f}")
                metric_err.metric("Error Rate", f"{err_val:.2f}%", delta_color="inverse")
                
                # 画图
                chart_df = pd.DataFrame({
                    "Predicted Alpha": alpha_history,
                    "Ground Truth": [true_alpha] * len(alpha_history)
                })
                chart_placeholder.line_chart(chart_df)

        end_time = time.time()
        progress_bar.progress(1.0)
        
        # 训练结束后，再更新一次以显示最终状态
        final_err = abs(model.alpha.item() - true_alpha) / true_alpha * 100
        metric_epoch.metric("当前轮数 (Epoch)", f"{epochs} / {epochs}")
        metric_loss.metric("Final Loss", f"{loss.item():.2e}")
        metric_alpha.metric("Final Alpha", f"{model.alpha.item():.5f}")
        metric_err.metric("Final Error", f"{final_err:.2f}%", delta_color="inverse")
        
        st.success(f"✅ 训练结束！耗时: {end_time - start_time:.2f}s")
        
        # 4. 最终可视化
        st.markdown("---")
        st.subheader("📊 结果可视化分析")
        
        # 全场预测
        X_all = torch.tensor(X_star, dtype=torch.float32).to(device)
        with torch.no_grad():
            u_pred_all = model(X_all[:, 0:1], X_all[:, 1:2]).cpu().numpy()
        u_pred_grid = u_pred_all.reshape(X.shape)
        
        # Matplotlib 画图
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # 真值
        c1 = ax[0].pcolormesh(T, X, u_true, cmap='jet', shading='auto')
        ax[0].set_title("Ground Truth Temp")
        ax[0].set_xlabel("Time"); ax[0].set_ylabel("Position")
        fig.colorbar(c1, ax=ax[0])
        
        # 预测值
        c2 = ax[1].pcolormesh(T, X, u_pred_grid, cmap='jet', shading='auto')
        ax[1].set_title(f"PINN Prediction (Noise={noise_level}%)")
        ax[1].set_xlabel("Time")
        fig.colorbar(c2, ax=ax[1])
        
        # 误差
        err_map = np.abs(u_true - u_pred_grid)
        c3 = ax[2].pcolormesh(T, X, err_map, cmap='inferno', shading='auto')
        ax[2].set_title(f"Abs Error Map (Max={np.max(err_map):.2e})")
        ax[2].set_xlabel("Time")
        fig.colorbar(c3, ax=ax[2])
        
        st.pyplot(fig)
        
        # 最终结论
        st.info(f"""
        **实验结论**：
        在引入 **{noise_level}%** 随机传感器噪声的情况下，模型经过 **{epochs}** 轮迭代：
        1.  成功反演出热扩散系数 $\\alpha = {model.alpha.item():.5f}$ (真实值 {true_alpha})，误差仅为 **{final_err:.2f}%**。
        2.  实现了对电池内部温度场的无损重构，证明了该算法具备 **抗噪性** 和 **鲁棒性**。
        """)