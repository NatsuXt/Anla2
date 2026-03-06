"""
visualize_ring_masking.py
=========================
Anla Ring Masking 实验的可视化与拓扑分析工具。

用途：
    1. 在训练过程中调用 RingMaskingVisualizer 记录动态数据并定期生成快照。
    2. 训练结束后调用 final_analysis() 生成完整的拓扑审计报告。

核心分析维度：
    - 训练动力学曲线（能量 / 准确率）
    - Embedding 拓扑结构（PCA / t-SNE 投影，按环位置着色）
    - 相位编码分析（逐维度相位热力图 + 相位线性度评分）
    - 最近邻链验证（embedding 空间中的邻居是否与环拓扑一致）

保存位置：Logs/ring_masking_vis/

使用方式（集成到训练脚本）：
    from Anla.visualization.visualize_ring_masking import RingMaskingVisualizer

    vis = RingMaskingVisualizer(base_dir=".", vocab_size=64)
    # 在训练循环内：
    vis.record(epoch, loss, acc)
    # 每隔一段或训练结束时：
    vis.snapshot(epoch, model.embedding, acc)
    # 训练完全结束后：
    vis.final_analysis(model.embedding)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ================= 加入以下两行全局配置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

class RingMaskingVisualizer:
    """
    Ring Masking 实验的可视化引擎。

    Parameters
    ----------
    base_dir : str
        项目根目录路径，日志将保存到 base_dir/Logs/ring_masking_vis/
    vocab_size : int
        词汇表大小（不含 mask token）。对于简单环，token i 的环位置就是 i。
    """

    def __init__(self, base_dir: str, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.log_dir = os.path.join(base_dir, "Logs", "ring_masking_vis")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[Vis] 日志目录: {self.log_dir}")

        # ---- 训练动态数据缓存 ----
        self.epochs = []
        self.losses = []
        self.accs = []

    # ==========================================================================
    # 公共接口
    # ==========================================================================

    def record(self, epoch: int, loss: float, acc: float):
        """每个 epoch 调用一次，记录训练动态。"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accs.append(acc)

    def snapshot(self, epoch: int, embedding: "ComplexEmbedding", acc: float):
        """
        在训练过程中定期调用，生成当前状态的快照图。
        包含：训练曲线 + PCA 拓扑 + 相位概览。

        Parameters
        ----------
        epoch : int
            当前 epoch 编号。
        embedding : ComplexEmbedding
            模型的 embedding 层（需要访问 .weight）。
        acc : float
            当前 batch 准确率。
        """
        weights = embedding.weight.detach().cpu()[:self.vocab_size]  # 排除 mask token
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        # ---- Panel 1: 训练动力学 ----
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_dynamics(ax1, acc)

        # ---- Panel 2: PCA 拓扑（按环位置着色）----
        ax2 = fig.add_subplot(gs[0, 1])
        coords_pca = self._pca_projection(weights)
        self._plot_ring_topology(ax2, coords_pca, title="PCA 投影")

        # ---- Panel 3: 模长分布 ----
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_magnitude_distribution(ax3, weights)

        # ---- Panel 4: 相位热力图（前 16 维）----
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_phase_heatmap(ax4, weights, max_dims=32)

        # ---- Panel 5: 相位线性度评分 ----
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_phase_linearity_scores(ax5, weights)

        fig.suptitle(
            f"Anla Ring Masking — Epoch {epoch} | Acc: {acc:.1%}",
            fontsize=14, fontweight="bold", y=0.98
        )
        save_path = os.path.join(self.log_dir, f"snapshot_epoch_{epoch:05d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def final_analysis(self, embedding: "ComplexEmbedding"):
        """
        训练结束后调用。生成完整的拓扑审计报告。
        包含以上所有分析 + t-SNE + 最近邻链 + 逐维度相位详情。

        Parameters
        ----------
        embedding : ComplexEmbedding
            训练完成后的 embedding 层。
        """
        print("[Vis] 正在生成最终分析报告...")
        weights = embedding.weight.detach().cpu()[:self.vocab_size]

        fig = plt.figure(figsize=(28, 20))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.35)

        # ===== 第一行：训练动力学 + 拓扑结构 =====

        # [1,1] 训练曲线
        ax_dyn = fig.add_subplot(gs[0, 0])
        self._plot_dynamics(ax_dyn, self.accs[-1] if self.accs else 0.0)

        # [1,2] PCA 拓扑
        ax_pca = fig.add_subplot(gs[0, 1])
        coords_pca = self._pca_projection(weights)
        self._plot_ring_topology(ax_pca, coords_pca, title="PCA 拓扑")

        # [1,3] t-SNE 拓扑
        ax_tsne = fig.add_subplot(gs[0, 2])
        try:
            coords_tsne = self._tsne_projection(weights)
            self._plot_ring_topology(ax_tsne, coords_tsne, title="t-SNE 拓扑")
        except Exception as e:
            ax_tsne.text(0.5, 0.5, f"t-SNE 失败:\n{e}",
                         ha="center", va="center", transform=ax_tsne.transAxes)
            ax_tsne.set_title("t-SNE 拓扑")

        # [1,4] 最近邻链分析
        ax_nn = fig.add_subplot(gs[0, 3])
        self._plot_nearest_neighbor_chain(ax_nn, weights)

        # ===== 第二行：相位分析 =====

        # [2,1:3] 相位热力图（全维度）
        ax_heat = fig.add_subplot(gs[1, :3])
        self._plot_phase_heatmap(ax_heat, weights, max_dims=self.vocab_size)

        # [2,4] 相位线性度评分
        ax_lin = fig.add_subplot(gs[1, 3])
        self._plot_phase_linearity_scores(ax_lin, weights)

        # ===== 第三行：深度诊断 =====

        # [3,1] 模长分布
        ax_mag = fig.add_subplot(gs[2, 0])
        self._plot_magnitude_distribution(ax_mag, weights)

        # [3,2] 最佳相位维度的详细散点
        ax_best = fig.add_subplot(gs[2, 1])
        self._plot_best_phase_dimension(ax_best, weights)

        # [3,3] embedding 复平面投影（选两个最佳维度）
        ax_cpx = fig.add_subplot(gs[2, 2])
        self._plot_complex_plane_projection(ax_cpx, weights)

        # [3,4] 邻居距离分布
        ax_dist = fig.add_subplot(gs[2, 3])
        self._plot_neighbor_distance_distribution(ax_dist, weights)

        fig.suptitle(
            "Anla Ring Masking — 最终拓扑审计报告",
            fontsize=16, fontweight="bold", y=0.99
        )
        save_path = os.path.join(self.log_dir, "FINAL_ANALYSIS.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Vis] 报告已保存: {save_path}")

        # ---- 额外输出文字摘要 ----
        self._print_summary(weights)

    # ==========================================================================
    # 内部绘图方法
    # ==========================================================================

    def _plot_dynamics(self, ax, current_acc: float):
        """训练动力学双轴图：左轴 loss（对数），右轴 accuracy。"""
        if not self.epochs:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                    transform=ax.transAxes)
            return

        color_loss = "#2563EB"
        color_acc = "#DC2626"

        ax.set_xlabel("Epoch")
        ax.set_ylabel("L_Elegant (log)", color=color_loss)
        ax.plot(self.epochs, self.losses, color=color_loss, linewidth=1.2, alpha=0.8)
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelcolor=color_loss)
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        ax2.set_ylabel("Accuracy", color=color_acc)
        ax2.plot(self.epochs, self.accs, color=color_acc, linewidth=1.2, alpha=0.8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis="y", labelcolor=color_acc)

        ax.set_title(f"训练动力学 (当前 Acc: {current_acc:.1%})")

    def _pca_projection(self, weights: torch.Tensor) -> np.ndarray:
        """将复数 embedding 权重投影到 2D PCA 空间。"""
        # 拼接实部虚部 → [vocab, 2*dim]
        flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        pca = PCA(n_components=2)
        return pca.fit_transform(flat)

    def _tsne_projection(self, weights: torch.Tensor) -> np.ndarray:
        """将复数 embedding 权重投影到 2D t-SNE 空间。"""
        flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        # perplexity 设高一些以适应环状结构
        perp = min(40, self.vocab_size - 1)
        tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto")
        return tsne.fit_transform(flat)

    def _plot_ring_topology(self, ax, coords: np.ndarray, title: str = ""):
        """
        在 2D 投影空间中绘制 embedding 点，按环位置（= token ID）着色。
        如果拓扑成功学习，应呈现彩虹色的闭合环。
        """
        # 对于简单环，环位置 = token ID
        ring_order = np.arange(self.vocab_size)

        # 绘制连接线（按环顺序）
        # 构造线段集合：token 0→1→2→...→63→0
        segments = []
        for i in range(self.vocab_size):
            j = (i + 1) % self.vocab_size
            segments.append([coords[i], coords[j]])
        lc = LineCollection(segments, colors="gray", linewidths=0.6, alpha=0.3)
        ax.add_collection(lc)

        # 散点（按环位置着色）
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=ring_order, cmap="hsv", s=40, alpha=0.9,
            edgecolors="white", linewidths=0.3, zorder=5
        )
        plt.colorbar(sc, ax=ax, label="Token ID (= 环位置)", shrink=0.8)

        # 标注几个关键点
        for label_id in [0, 16, 32, 48]:
            if label_id < self.vocab_size:
                ax.annotate(
                    str(label_id),
                    xy=(coords[label_id, 0], coords[label_id, 1]),
                    fontsize=7, fontweight="bold", color="black",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7)
                )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")

    def _plot_phase_heatmap(self, ax, weights: torch.Tensor, max_dims: int = 32):
        """
        相位热力图：x 轴 = token ID（按环顺序），y 轴 = 维度索引。
        颜色 = angle(embedding[token, dim])，范围 [-π, π]。

        如果相位在编码环位置，某些维度应呈现从左到右平滑变化的条纹。
        """
        phases = torch.angle(weights).numpy()  # [vocab, dim]
        d_model = phases.shape[1]
        show_dims = min(d_model, max_dims)

        # 按照"对环位置的线性相关度"排序维度（最相关的在最上面）
        linearity = self._compute_phase_linearity(weights)
        sorted_dims = np.argsort(-linearity)[:show_dims]
        phases_sorted = phases[:, sorted_dims].T  # [show_dims, vocab]

        im = ax.imshow(
            phases_sorted, aspect="auto", cmap="twilight",
            vmin=-np.pi, vmax=np.pi, interpolation="nearest"
        )
        plt.colorbar(im, ax=ax, label="Phase (rad)", shrink=0.8)

        ax.set_xlabel("Token ID (环顺序)")
        ax.set_ylabel(f"维度索引 (按线性度排序, 前 {show_dims})")
        ax.set_title("相位热力图 (twilight: 相位连续 = 条纹)")

        # y 轴标签：显示原始维度编号
        if show_dims <= 32:
            ax.set_yticks(range(show_dims))
            ax.set_yticklabels([f"d{sorted_dims[i]}" for i in range(show_dims)],
                               fontsize=5)

    def _compute_phase_linearity(self, weights: torch.Tensor) -> np.ndarray:
        """
        对每个维度计算：该维度的相位与环位置的圆形相关系数 (circular correlation)。

        方法：
            对于维度 d，令 φ_k = angle(w[k, d])，环位置 θ_k = 2π·k/N。
            圆形相关 = |mean(exp(i·(φ_k - α·θ_k)))| 对 α 取最大值。
            简化近似：使用 |Σ exp(i·φ_k) · exp(-i·θ_k)| / N。

        返回
        ------
        linearity : ndarray, shape [dim]
            每个维度的相位线性度得分，范围 [0, 1]。
        """
        phases = torch.angle(weights).numpy()  # [vocab, dim]
        N = self.vocab_size
        d_model = phases.shape[1]

        # 环位置对应的参考相位（均匀分布在 [0, 2π)）
        ref_angles = 2.0 * np.pi * np.arange(N) / N  # [N]

        scores = np.zeros(d_model)
        # 尝试多个频率倍数（环可能被编码为 1x, 2x, 3x 频率）
        for freq in [1, 2, 3, 4]:
            ref = np.exp(1j * freq * ref_angles)  # [N]
            sig = np.exp(1j * phases)              # [N, dim]
            # 逐维度计算圆形互相关
            corr = np.abs(np.mean(sig * ref[:, None].conj(), axis=0))
            scores = np.maximum(scores, corr)

        return scores

    def _plot_phase_linearity_scores(self, ax, weights: torch.Tensor):
        """柱状图：每个维度的相位线性度得分。"""
        scores = self._compute_phase_linearity(weights)
        d_model = len(scores)
        sorted_idx = np.argsort(-scores)

        colors = plt.cm.RdYlGn(scores[sorted_idx] / max(scores.max(), 1e-9))
        ax.barh(range(d_model), scores[sorted_idx], color=colors, height=0.8)
        ax.set_xlabel("圆形相关系数")
        ax.set_ylabel("维度 (按得分降序)")
        ax.set_title("相位线性度评分")
        ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="阈值 0.3")
        ax.legend(fontsize=7)

        # 只标注前几个
        for i in range(min(5, d_model)):
            ax.text(scores[sorted_idx[i]] + 0.01, i,
                    f"d{sorted_idx[i]}: {scores[sorted_idx[i]]:.3f}",
                    fontsize=6, va="center")

        ax.invert_yaxis()
        ax.set_xlim(0, max(scores.max() * 1.2, 0.5))

    def _plot_magnitude_distribution(self, ax, weights: torch.Tensor):
        """Embedding 向量模长的分布直方图。"""
        # 逐 token 的 RMS 模长: [vocab]
        mags = weights.abs().pow(2).mean(dim=1).sqrt().numpy()

        # 检查数据范围，防止数值崩溃
        mag_min, mag_max = np.min(mags), np.max(mags)
        data_range = mag_max - mag_min
        
        # 如果所有模长几乎相等（例如初始化阶段），将 bins 设为 1 或指定固定 range
        if data_range < 1e-6:
            # 此时数据几乎是一个点，手动指定 range 避免 numpy 计算错误
            hist_bins = 10
            hist_range = (mag_min - 0.1, mag_max + 0.1)
        else:
            hist_bins = 30
            hist_range = (mag_min, mag_max)

        ax.hist(
            mags, 
            bins=hist_bins, 
            range=hist_range,
            color="#6366F1", 
            alpha=0.8, 
            edgecolor="white"
        )
        
        ax.axvline(x=np.mean(mags), color="red", linestyle="--",
                   label=f"均值: {np.mean(mags):.4f}")
        ax.axvline(x=1.0, color="black", linestyle=":", alpha=0.5, label="目标: 1.0")
        
        ax.set_xlabel("|w| (RMS 模长)")
        ax.set_ylabel("Token 数量")
        ax.set_title("Embedding 模长分布")
        ax.legend(fontsize=7)

    def _plot_best_phase_dimension(self, ax, weights: torch.Tensor):
        """找到相位线性度最高的维度，绘制其相位 vs token ID 的散点图。"""
        scores = self._compute_phase_linearity(weights)
        best_dim = int(np.argmax(scores))
        phases = torch.angle(weights[:, best_dim]).numpy()

        ring_order = np.arange(self.vocab_size)
        ax.scatter(ring_order, phases, c=ring_order, cmap="hsv", s=20, alpha=0.8)
        ax.set_xlabel("Token ID (= 环位置)")
        ax.set_ylabel("Phase (rad)")
        ax.set_title(f"最佳相位维度 d{best_dim} (分数: {scores[best_dim]:.3f})")
        ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
        ax.axhline(y=np.pi, color="gray", linestyle=":", alpha=0.3)
        ax.axhline(y=-np.pi, color="gray", linestyle=":", alpha=0.3)
        ax.grid(True, alpha=0.2)

    def _plot_complex_plane_projection(self, ax, weights: torch.Tensor):
        """选取相位线性度最高的一个维度，在复平面上画出所有 token 的 embedding 值。"""
        scores = self._compute_phase_linearity(weights)
        best_dim = int(np.argmax(scores))
        z = weights[:, best_dim].numpy()  # complex array [vocab]

        ring_order = np.arange(self.vocab_size)
        ax.scatter(z.real, z.imag, c=ring_order, cmap="hsv", s=25, alpha=0.85,
                   edgecolors="white", linewidths=0.3)

        # 绘制环连接线
        for i in range(self.vocab_size):
            j = (i + 1) % self.vocab_size
            ax.plot([z[i].real, z[j].real], [z[i].imag, z[j].imag],
                    color="gray", linewidth=0.4, alpha=0.3)

        # 画单位圆参考
        theta = np.linspace(0, 2 * np.pi, 100)
        r_mean = np.mean(np.abs(z))
        ax.plot(r_mean * np.cos(theta), r_mean * np.sin(theta),
                color="black", linestyle=":", alpha=0.3, linewidth=0.8)

        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"复平面 (维度 d{best_dim})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

    def _plot_nearest_neighbor_chain(self, ax, weights: torch.Tensor):
        """
        最近邻链分析：对每个 token，在 embedding 空间中找最近邻，
        然后统计"最近邻与环上邻居的一致率"。

        如果学到了环拓扑，token i 的最近邻应该是 (i-1) 或 (i+1) mod N。
        """
        # 计算两两距离矩阵
        # [V, dim] 复数差的模
        w = weights  # [V, dim]
        # dist[i,j] = ||w[i] - w[j]||^2
        diff = w.unsqueeze(0) - w.unsqueeze(1)  # [V, V, dim]
        dists = diff.abs().pow(2).sum(dim=2)     # [V, V]

        # 自身距离设为 inf
        dists.fill_diagonal_(float("inf"))

        # 最近邻
        nn_ids = dists.argmin(dim=1).numpy()  # [V]
        token_ids = np.arange(self.vocab_size)

        # 检查最近邻是否是环上的 +1 或 -1 邻居
        prev_neighbor = (token_ids - 1) % self.vocab_size
        next_neighbor = (token_ids + 1) % self.vocab_size

        is_ring_neighbor = (nn_ids == prev_neighbor) | (nn_ids == next_neighbor)
        ring_nn_rate = is_ring_neighbor.mean()

        # 绘制：x = token ID, y = 最近邻 ID，颜色 = 是否为环邻居
        colors = ["#22C55E" if ok else "#EF4444" for ok in is_ring_neighbor]
        ax.scatter(token_ids, nn_ids, c=colors, s=20, alpha=0.8, zorder=5)

        # 画理想的 +1 线和 -1 线
        ax.plot(token_ids, next_neighbor, color="blue", linestyle="--",
                alpha=0.3, linewidth=0.8, label="+1 邻居")
        ax.plot(token_ids, prev_neighbor, color="orange", linestyle="--",
                alpha=0.3, linewidth=0.8, label="-1 邻居")

        ax.set_xlabel("Token ID")
        ax.set_ylabel("最近邻 Token ID")
        ax.set_title(f"最近邻链 (环邻居一致率: {ring_nn_rate:.1%})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.15)

    def _plot_neighbor_distance_distribution(self, ax, weights: torch.Tensor):
        """
        对比两类距离的分布：
        1. 环上相邻 token 之间的 embedding 距离（应该小）
        2. 环上距离为 N/2 的 token 之间的 embedding 距离（应该大）

        如果两个分布分离良好，说明 embedding 几何忠实反映了环拓扑。
        """
        w = weights.numpy()  # complex [V, dim]

        # 相邻距离
        adj_dists = []
        for i in range(self.vocab_size):
            j = (i + 1) % self.vocab_size
            d = np.sqrt(np.sum(np.abs(w[i] - w[j]) ** 2))
            adj_dists.append(d)

        # 对径距离（环上距离 = N/2）
        half = self.vocab_size // 2
        opp_dists = []
        for i in range(self.vocab_size):
            j = (i + half) % self.vocab_size
            d = np.sqrt(np.sum(np.abs(w[i] - w[j]) ** 2))
            opp_dists.append(d)

        ax.hist(adj_dists, bins=20, alpha=0.7, color="#3B82F6",
                label=f"相邻 (μ={np.mean(adj_dists):.3f})", density=True)
        ax.hist(opp_dists, bins=20, alpha=0.7, color="#F97316",
                label=f"对径 (μ={np.mean(opp_dists):.3f})", density=True)
        ax.set_xlabel("Embedding 距离")
        ax.set_ylabel("密度")
        ax.set_title("距离分布：相邻 vs 对径")
        ax.legend(fontsize=7)

    # ==========================================================================
    # 文字摘要
    # ==========================================================================

    def _print_summary(self, weights: torch.Tensor):
        """打印一份简洁的文字摘要到终端。"""
        scores = self._compute_phase_linearity(weights)
        mags = weights.abs().pow(2).mean(dim=1).sqrt().numpy()

        # 最近邻链统计
        w = weights
        diff = w.unsqueeze(0) - w.unsqueeze(1)
        dists = diff.abs().pow(2).sum(dim=2)
        dists.fill_diagonal_(float("inf"))
        nn_ids = dists.argmin(dim=1).numpy()
        token_ids = np.arange(self.vocab_size)
        prev_n = (token_ids - 1) % self.vocab_size
        next_n = (token_ids + 1) % self.vocab_size
        ring_nn_rate = ((nn_ids == prev_n) | (nn_ids == next_n)).mean()

        # 高线性度维度数
        high_lin_count = int(np.sum(scores > 0.3))

        print("\n" + "=" * 60)
        print(" Anla Ring Masking — 拓扑审计摘要")
        print("=" * 60)
        print(f"  Embedding 模长:  均值 {np.mean(mags):.4f}, "
              f"范围 [{np.min(mags):.4f}, {np.max(mags):.4f}]")
        print(f"  最近邻环一致率:  {ring_nn_rate:.1%}")
        print(f"  高线性度维度数:  {high_lin_count}/{len(scores)} "
              f"(阈值 > 0.3)")
        print(f"  最佳维度:        d{np.argmax(scores)} "
              f"(分数: {scores.max():.4f})")
        if self.losses:
            print(f"  最终 L_Elegant:   {self.losses[-1]:.6f}")
        if self.accs:
            print(f"  最终准确率:      {self.accs[-1]:.1%}")
        print("=" * 60)

        # 判定
        if ring_nn_rate > 0.9 and high_lin_count >= 3:
            print("  [✓] 拓扑判定: 环结构已被相位忠实编码。")
        elif ring_nn_rate > 0.7:
            print("  [~] 拓扑判定: 环结构部分形成，但相位编码不完全。")
        else:
            print("  [✗] 拓扑判定: 环结构未在 embedding 中形成。")
        print() 


# ==============================================================================
# 独立运行入口：加载 checkpoint 并生成分析报告
# ==============================================================================

def analyze_checkpoint(checkpoint_path: str, vocab_size: int = 64, d_model: int = 64):
    """
    从保存的 checkpoint 加载模型，生成完整分析报告。

    用法:
        python -m Anla.visualization.visualize_ring_masking --checkpoint checkpoints/best_ring_model.pth
    """
    import sys
    # [Path Fix] 文件位置: Anla/visualization/visualize_ring_masking.py
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    _ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, ".."))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, ".."))
    sys.path.insert(0, _PROJECT_ROOT)
    from Anla.layers.embedding import ComplexEmbedding

    print(f"[Vis] 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 重建 embedding 层并加载权重
    embed = ComplexEmbedding(vocab_size + 1, d_model)

    # 从 state_dict 中提取 embedding 权重
    state = ckpt.get("model_state_dict", ckpt)
    embed_key = None
    for k in state:
        if "embedding.weight" in k or "embed.weight" in k:
            embed_key = k
            break

    if embed_key is None:
        print("[Vis] 错误: 无法在 checkpoint 中找到 embedding 权重。")
        print(f"      可用的 keys: {list(state.keys())[:10]}...")
        return

    embed.weight.data = state[embed_key]
    print(f"[Vis] Embedding 权重已加载: {embed.weight.shape}")

    vis = RingMaskingVisualizer(
        base_dir=os.path.dirname(os.path.dirname(checkpoint_path)) or ".",
        vocab_size=vocab_size
    )
    vis.final_analysis(embed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anla Ring Masking 可视化分析")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_ring_model.pth",
        help="checkpoint 文件路径"
    )
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint, args.vocab_size, args.d_model)
