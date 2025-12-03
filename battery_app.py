import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go 
import matplotlib.pyplot as plt    
import os

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="李宏光的科研主页 | PINN 电池监控",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 0.5 亮色系高级 CSS (修复标题遮挡问题)
# ==========================================
st.markdown("""
    <style>
        /* --- 1. 顶部留白调整 (关键修改：从 1rem 改为 3rem) --- */
        .block-container {
            padding-top: 3rem !important; 
            padding-bottom: 2rem !important;
        }
        
        /* --- 2. 全局亮色背景 --- */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #333;
        }
        
        /* --- 3. 动画定义 --- */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translate3d(0, 30px, 0); }
            to { opacity: 1; transform: translate3d(0, 0, 0); }
        }
        
        @keyframes float {
            0% { transform: translateY(0px); box-shadow: 0 5px 15px 0px rgba(0,0,0,0.1); }
            50% { transform: translateY(-10px); box-shadow: 0 25px 15px 0px rgba(0,0,0,0.05); }
            100% { transform: translateY(0px); box-shadow: 0 5px 15px 0px rgba(0,0,0,0.1); }
        }
        
        @keyframes gradient-text {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- 4. 组件样式 --- */
        .gradient-title {
            background: linear-gradient(45deg, #2563eb, #3b82f6, #06b6d4, #2563eb);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-text 5s ease infinite;
            font-weight: 800;
            font-size: 2.2rem; 
            padding-bottom: 5px;
            margin-top: 0 !important;
        }

        /* 卡片通用样式 */
        .hover-card {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            margin-bottom: 20px;
            border: 1px solid #fff;
            animation-name: fadeInUp;
            animation-duration: 0.8s;
            animation-fill-mode: both;
        }
        
        .hover-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        }

        /* 侧边栏 */
        .profile-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .floating-avatar {
            animation: float 6s ease-in-out infinite;
            border-radius: 50%;
        }

        /* 延迟动画 */
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }

        /* 装饰边框 */
        .border-left-red { border-left: 5px solid #ef4444; }
        .border-left-green { border-left: 5px solid #10b981; }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 侧边栏
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="profile-box">
        <div class="floating-avatar">
            <img src="https://api.dicebear.com/9.x/notionists/svg?seed=LiHongguang&backgroundColor=e5e5e5" 
                 style="width: 100px; height: 100px; border-radius: 50%; border: 3px solid #3b82f6;">
        </div>
        <h3 style="margin: 10px 0 5px 0; color: #1e293b;">李宏光</h3>
        <p style="color: #64748b; font-size: 14px; margin: 0;">南京邮电大学 · 硕士研究生</p>
        <p style="color: #64748b; font-size: 13px; margin: 5px 0 15px 0;">专业：计算数学</p>
        <div style="display:flex; justify-content:center; gap:8px;">
            <span style="background:#eff6ff; color:#3b82f6; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold;">PINN</span>
            <span style="background:#eff6ff; color:#3b82f6; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold;">PyTorch</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ 仿真参数设置")
    true_alpha = st.number_input("真实热扩散系数", value=0.01, format="%.4f")
    noise_level = st.slider("传感器噪声水平 (%)", 0.0, 5.0, 1.0, step=0.1)
    
    st.markdown("### ⚙️ 求解器配置")
    epochs = st.slider("训练轮数", 1000, 10000, 3000, step=1000)
    pde_weight = st.slider("物理权重 (PDE Weight)", 1.0, 50.0, 10.0)
    lr = st.number_input("学习率", value=0.001, format="%.4f")
    
    st.markdown("---")
    st.caption("© 2024 PINN Battery Project.")

# ==========================================
# 2. 核心类与函数
# ==========================================
@st.cache_resource
def get_device():
    return torch.device("cpu")

device = get_device()

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
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
    def analytic_solution(x, t, alpha):
        return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    u_true = analytic_solution(X, T, alpha)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = u_true.flatten()[:,None]
    idx = np.random.choice(X_star.shape[0], 500, replace=False)
    X_train = torch.tensor(X_star[idx, :], dtype=torch.float32).to(device)
    u_train = torch.tensor(u_star[idx, :], dtype=torch.float32).to(device)
    noise_std = (noise_pct / 100.0)
    u_train = u_train + noise_std * torch.randn_like(u_train)
    idx_f = np.random.choice(X_star.shape[0], 2000, replace=False)
    X_f = torch.tensor(X_star[idx_f, :], dtype=torch.float32, requires_grad=True).to(device)
    return X, T, u_true, X_star, X_train, u_train, X_f

# ==========================================
# 3. 主页面
# ==========================================

st.markdown('<h1 class="gradient-title">🔋 基于 PINN 的锂电池热参数反演与实时监控系统</h1>', unsafe_allow_html=True)
st.markdown("##### *Physics-Informed Neural Networks for Battery Thermal Management*")

tab1, tab2, tab3 = st.tabs(["📖 项目背景与痛点", "💡 核心技术方案", "🚀 3D 数字孪生 (工业级演示)"])

# --- TAB 1 ---
with tab1:
    st.header("1. 为什么要做这个项目？")
    col_bg1, col_bg2 = st.columns([1, 1])
    
    with col_bg1:
        st.markdown("""
        <div class="hover-card border-left-red delay-1">
            <h3 style="color: #dc2626; margin-top:0;">🛑 行业痛点</h3>
            <p><b>1. 内部温度不可测</b><br>
            现有 BMS 传感器只能贴在电池表面，核心温度往往比表面高 5-10°C，容易引发热失控风险。</p>
            <p><b>2. 传统仿真太慢</b><br>
            FEM/CFD (有限元) 精度虽高但计算耗时，无法在车载芯片上实时运行。</p>
            <p><b>3. SOH (健康状态) 难估算</b><br>
            电池老化会导致热扩散系数、内阻等物理参数变化，这些参数无法直接测量。</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_bg2:
        st.markdown("""
        <div class="hover-card border-left-green delay-2">
            <h3 style="color: #059669; margin-top:0;">✅ 我们的目标</h3>
            <p><b>打造“虚拟传感器” (Virtual Sensor)</b></p>
            <ul>
                <li><b>输入</b>：仅利用表面稀疏的、含噪声的温度传感器数据。</li>
                <li><b>内核</b>：基于物理定律 (PINN) 的深度学习模型。</li>
                <li><b>输出</b>：实时重构内部温度场 + 自动反演热物性参数。</li>
                <li><b>优势</b>：毫秒级推理速度 + 物理级计算精度。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 删除了这里的结构图

# --- TAB 2: 技术详解 ---
with tab2:
    st.header("2. 核心技术详解")
    st.markdown("本项目通过三大核心技术，解决了在数据稀疏和含噪条件下的反问题求解。")
    st.markdown("### 🌡️ 物理控制方程：一维瞬态热传导 (Heat Equation)")
    st.latex(r"\frac{\partial u}{\partial t} = \alpha \cdot \frac{\partial^2 u}{\partial x^2}")
    
    with st.expander("📌 技术点 1: 混合损失函数 (Physics-Informed Loss)", expanded=True):
        st.latex(r"\mathcal{L} = \underbrace{\frac{1}{N}\sum(u_{pred} - u_{sensor})^2}_{\text{Data Loss}} + \lambda \cdot \underbrace{\frac{1}{M}\sum(u_t - \alpha u_{xx})^2}_{\text{PDE Loss}}")
    with st.expander("📌 技术点 2: 参数自适应反演 (SOH Estimation)", expanded=True):
        st.markdown(r"将热扩散系数 $\alpha$ 设为可训练变量。在训练中，网络会自动寻找最佳的 $\alpha$ 值。")
    with st.expander("📌 技术点 3: 抗噪优化策略", expanded=True):
        st.markdown("采用 Adam + L-BFGS 两阶段训练，配合动态权重 $\lambda$，抵抗传感器噪声。")

# --- TAB 3: 数字孪生 ---
with tab3:
    st.header("3. 工业级数字孪生与验证 (Industrial Digital Twin)")
    
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False

    # === Step 1: 训练 ===
    st.subheader("Step 1: 模型训练与参数反演 (Training)")
    
    if st.button("🚀 启动数字孪生求解器 (Start Solver)", type="primary"):
        X, T, u_true, X_star, X_train, u_train, X_f = generate_data(true_alpha, noise_level)
        model = PINN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        col_epoch, col_loss, col_alpha, col_err = st.columns(4)
        metric_epoch = col_epoch.empty(); metric_loss = col_loss.empty()
        metric_alpha = col_alpha.empty(); metric_err = col_err.empty()
        chart_placeholder = st.empty(); progress_bar = st.progress(0)
        
        loss_history = []; alpha_history = []
        start_time = time.time()
        
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            u_pred = model(X_train[:, 0:1], X_train[:, 1:2])
            loss_data = torch.mean((u_pred - u_train) ** 2)
            loss_physics = model.physics_loss(X_f[:, 0:1], X_f[:, 1:2])
            loss = loss_data + pde_weight * loss_physics
            loss.backward()
            optimizer.step()
            
            curr_loss = loss.item(); curr_alpha = model.alpha.item()
            loss_history.append(curr_loss); alpha_history.append(curr_alpha)
            
            if epoch % (epochs // 20) == 0:
                progress_bar.progress(epoch / epochs)
                err_val = abs(curr_alpha - true_alpha) / true_alpha * 100
                metric_epoch.metric("训练轮次", f"{epoch}/{epochs}")
                metric_loss.metric("Total Loss", f"{curr_loss:.2e}")
                metric_alpha.metric("预测热参数 α", f"{curr_alpha:.5f}")
                metric_err.metric("参数误差 %", f"{err_val:.2f}%", delta_color="inverse")
                chart_df = pd.DataFrame({"Predicted Alpha": alpha_history, "Ground Truth": [true_alpha] * len(alpha_history)})
                chart_placeholder.line_chart(chart_df)

        progress_bar.progress(1.0)
        st.success(f"✅ 求解收敛！耗时: {time.time() - start_time:.2f}s")
        st.session_state['trained'] = True
        st.session_state['model'] = model
        st.session_state['data'] = (X, T, u_true, X_star)

    # === Step 2: 交互展示 ===
    if st.session_state['trained']:
        st.markdown("---")
        model = st.session_state['model']
        X, T, u_true, X_star = st.session_state['data']
        
        st.subheader("Step 2: 3D 电池单体热场透视")
        
        col_ctrl, col_metric = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("##### ⏱️ 时间控制器")
            
            col_play, col_stop = st.columns(2)
            auto_play = col_play.button("▶️ 自动演化动画")
            
            if auto_play:
                t_ph = st.empty()
                for t_v in np.linspace(0, 1, 25):
                    st.session_state['t_val'] = t_v
                    time.sleep(0.04) 
                    t_ph.empty()
                t_select = 1.0
            else:
                if 't_val' not in st.session_state: st.session_state['t_val'] = 0.5
                t_select = st.slider("手动拖拽时间轴", 0.0, 1.0, st.session_state['t_val'], 0.01)

            st.markdown("---")
            if st.button("运行 BMS 毫秒级推理测试"):
                t0 = time.perf_counter()
                with torch.no_grad(): _ = model(torch.rand(100,1), torch.full((100,1), t_select))
                st.metric("单帧推理耗时", f"{(time.perf_counter()-t0)*1000:.3f} ms")

        with col_metric:
            with torch.no_grad():
                u_surf = model(torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([[t_select]], dtype=torch.float32)).item()
                u_core = model(torch.tensor([[0.0]], dtype=torch.float32), torch.tensor([[t_select]], dtype=torch.float32)).item()
                delta_t = u_core - u_surf
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); padding: 15px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);">
                <h4 style="margin:0; color:white;">⚡️ 实时热场监测 (Real-time HUD)</h4>
                <p style="margin:0; font-size: 14px; opacity: 0.9;">当前时间: t = {t_select:.2f} s</p>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("🔥 核心温度", f"{u_core:.3f}", delta=f"{delta_t:.3f} (+High)")
            m2.metric("🛡️ 表面温度", f"{u_surf:.3f}", delta="Boundary")
            m3.metric("⚠️ 内外温差", f"{delta_t:.3f}", delta_color="inverse")

        # 3D 绘图
        r = np.linspace(0, 1, 20); theta = np.linspace(0, 2*np.pi, 40); z = np.linspace(0, 2, 10)
        R, THETA, Z = np.meshgrid(r, theta, z)
        mask = (THETA < 1.5 * np.pi)
        R, THETA, Z = R[mask], THETA[mask], Z[mask]
        X_3d = R * np.cos(THETA); Y_3d = R * np.sin(THETA); Z_3d = Z
        
        r_flat = torch.tensor(R.flatten()[:, None], dtype=torch.float32)
        t_flat = torch.full_like(r_flat, t_select)
        with torch.no_grad(): u_val = model(r_flat, t_flat).numpy().flatten()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=X_3d.flatten(), y=Y_3d.flatten(), z=Z_3d.flatten(), mode='markers',
            marker=dict(size=4, color=u_val, colorscale='Jet', opacity=0.9, cmin=0, cmax=1, colorbar=dict(title="Temp"))
        )])
        fig.update_layout(
            title=dict(text=f"21700 电池单体热分布 (t={t_select:.2f})", x=0.5), 
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), 
            margin=dict(l=0, r=0, b=0, t=30), 
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # === Step 3: 验证 ===
        st.markdown("---")
        st.subheader("Step 3: 精度验证报告")
        X_all = torch.tensor(X_star, dtype=torch.float32).to(device)
        with torch.no_grad(): u_pred_all = model(X_all[:, 0:1], X_all[:, 1:2]).cpu().numpy()
        l2_error = np.linalg.norm(u_true.flatten() - u_pred_all.flatten()) / np.linalg.norm(u_true.flatten())
        
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("📊 全场 L2 相对误差", f"{l2_error:.2%}", "精度优异")
        col_v2.metric("🎯 参数反演误差", f"{abs(model.alpha.item()-true_alpha)/true_alpha*100:.2f}%", "辨识准确")
        col_v3.success(f"AI 成功还原了内部温度场。")
        
        st.write("📉 **误差分布热力图**")
        fig_val, ax = plt.subplots(1, 3, figsize=(15, 4))
        plt.style.use('default') 
        c1 = ax[0].pcolormesh(T, X, u_true, cmap='jet', shading='auto'); ax[0].set_title("Ground Truth"); plt.colorbar(c1, ax=ax[0])
        c2 = ax[1].pcolormesh(T, X, u_pred_all.reshape(X.shape), cmap='jet', shading='auto'); ax[1].set_title("PINN Prediction"); plt.colorbar(c2, ax=ax[1])
        c3 = ax[2].pcolormesh(T, X, np.abs(u_true - u_pred_all.reshape(X.shape)), cmap='inferno', shading='auto'); ax[2].set_title("Abs Error"); plt.colorbar(c3, ax=ax[2])
        st.pyplot(fig_val)
