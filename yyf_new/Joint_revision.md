
这是一个数据集采集的任务的文章书写工作，初版的latex版本如下
我们的工作就是查找如下初版中每个板块，对应到的代码区域，然后检查这个初版中涉及的部分是否书写符合实际，是否缺失了该板块的核心任务，表达是否学术化，规范
请你按照我给的代码板块顺序检查和修改
每次只改一个，改完后等待我审阅完才可以改下一个。

你可以先制定一个计划，我检查这个执行计划是否合理

---------------------------------------------------------
下面是文稿初版（latex）

%%
%% This is file `sample-sigconf.tex'修正版
%%
\documentclass[sigconf,screen,anonymous,review]{acmart}

% --- 1. 核心宏包：提供数学公式和符号支持 ---
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{bm} % 用于加粗数学符号
\usepackage{graphicx}

\renewcommand\footnotetextcopyrightpermission[1]{} % 移除底部版权
\settopmatter{printacmref=false} % 移除 ACM 引用格式

\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}

% 会议信息设定
\setcopyright{acmlicensed}
\copyrightyear{2026}
\acmYear{2026}
\acmDOI{XXXXXXX.XXXXXXX}
\acmConference[MM '26]{Proceedings of the 34th ACM International Conference on Multimedia}{November 10--14, 2026}{Rio de Janeiro, Brazil}
\acmISBN{978-1-4503-XXXX-X/2018/06}

\begin{document}

\title{Supplementary Material: Hardware-Aligned Capture System and Methodology}

% --- 作者信息 ---
\author{Ben Trovato}
\authornote{Both authors contributed equally to this research.}
\email{trovato@corporation.com}
\affiliation{%
  \institution{Institute for Clarity in Documentation}
  \city{Dublin}
  \state{Ohio}
  \country{USA}
}

\renewcommand{\shortauthors}{Trovato et al.}

\begin{abstract}
This supplementary material provides detailed information regarding the hardware-aligned capture system used for constructing our real-world benchmark. It describes the physical synchronization mechanisms, robust temporal post-alignment strategies, and highly accurate spatial alignment procedures that ensure the reliability of our real-world evaluation data.
\end{abstract}

\maketitle

% --- 正文部分 ---

\section{Hardware-Aligned Capture System and Methodology}
\label{sec:methodology}

\subsection{Overview}
To thoroughly evaluate the generalization capability of our Com\-ple\-men\-tary-Path\-way Distillation framework in real physical environments, we designed and constructed a high-precision hardware-aligned capture system. This system is composed of two sensors: a \textbf{CVS}, serving as the main vision sensor to provide complementary modality signals, and an \textbf{Intel RealSense D455} depth camera, acting as the secondary sensor to provide ground-truth depth maps $\mathcal{D}_{gt}$. The CVS delivers two complementary pathways: a slow cognition-oriented pathway $\mathcal{C}$ operating at approximately 30\,Hz, and a high-speed action-oriented pathway $\mathcal{SD\&TD}$ that encodes both spatial differences (SD) and temporal differences (TD) at a significantly higher temporal resolution.

% ---  强制缩放表格防止超限 ---
\begin{table}[h]
\caption{Sensor Specifications}
\label{tab:sensors}
\resizebox{\columnwidth}{!}{% 强制表格宽度等于分栏宽度
\begin{tabular}{l|cc}
\hline
\textbf{Feature} & \textbf{CVS} & \textbf{RealSense D455} \\ \hline
Pathway & $\mathcal{C}$ + $\mathcal{SD\&TD}$ & Color + Depth (${D}_{gt}$) \\
Resolution & $640 \times 320$ & $640 \times 480$ (Cropped) \\
Frame Rate & $\sim$30.3\,Hz (Master) & 60\,FPS (Slave, Genlock) \\
Shutter Type & Global Shutter & Global Shutter \\
Sync Method & GPIO (Master) & External Trigger (Slave) \\ \hline
\end{tabular}
}
\end{table}

The fundamental challenge in fusing such sensors lies in resolving their spatiotemporal mismatch. Specifically, the system must achieve high-accuracy temporal synchronization to avoid motion-induced errors and high-precision spatial alignment to ensure pixel-level consistency. To address these challenges, we developed a hardware-aligned capture system coupled with dedicated software post-processing, establishing a robust paradigm that guarantees the reliability and scientific rigor of our real-world evaluation benchmark.

\subsection{Hardware-Level Temporal Synchronization}
Ensuring physical synchronization between the two different sensors is a prerequisite for accurate depth estimation, particularly under high-speed motion dynamics. To achieve this, we employ a \textbf{Master-Slave hardware architecture} controlled by an NVIDIA Jetson Orin platform.

The CVS event camera acts as the temporal master. Once powered on via a GPIO pin (Pin 15) raised by the Jetson Orin, the CVS camera continuously emits external trigger pulses at approximately 30.3\,Hz, each synchronized to the onset of a $\mathcal{C}$-stream exposure. Simultaneously, the RealSense D455 is configured in Genlock mode (\texttt{RS2\_OPTION\_INTER\_CAM\_SYNC\_MODE}=4) as a temporal slave, receiving these trigger pulses to align its exposure window. To ensure the RealSense reports raw hardware timestamps without software time compensation, we further disable \texttt{RS2\_OPTION\_GLOBAL\_TIME\_ENABLED} for both the Stereo Module and the RGB camera. This hardware-level trigger mechanism establishes the physical synchronization foundation: every RealSense frame exposure is temporally locked to a CVS frame exposure.

To verify this alignment, the Jetson Orin simultaneously launches the RealSense recording thread and a timestamp polling thread. The polling thread repeatedly queries the CVS camera's FPGA---via USB at the moment the \texttt{main} process is forked---until it receives a non-zero Unix timestamp. Because the polled timestamp and the timestamps embedded in every CVS frame originate from the same FPGA clock, the first non-zero value $T_{ref}$ returned by the FPGA is guaranteed to equal the timestamp $t_n$ of the CVS frame that is active at the exact moment the RealSense starts recording. This gives:
\begin{equation}
    |t_{\hat{n}} - T_{ref}| = 0,
\end{equation}
which we denote as the initially aligned frame $\hat{n}$:
\begin{equation}
    \hat{n} = \arg\min_{n} |t_n - T_{ref}|
\end{equation}
Consequently, frame $\hat{n}$ of the $\mathcal{C}$-stream and the first frame of the RealSense depth stream share a common temporal origin, providing a hardware-verified zero-error temporal anchor.

\subsection{Robust Temporal Post-Alignment}
While hardware synchronization aligns the physical exposure, transmission delays and system buffer dynamics can still introduce temporal perturbations. To ensure strict temporal linearity across all three streams (CVS $\mathcal{C}$, $\mathcal{SD\&TD}$, and RealSense depth $\mathcal{D}_{gt}$), we perform a two-stage temporal post-processing pipeline.

\paragraph{Anomaly Compensation and Frame Filtering}
Subsequently, we implement a robust verification mechanism based on inter-frame intervals $\Delta t = t_i - t_{i-1}$. By continuously monitoring $\Delta t$, the system automatically detects two types of transmission anomalies:

\begin{itemize}
    \item \textbf{Stationary duplicates:} When $\Delta t \approx 0$\,ms (indicating no actual motion capture), the frame is retained but no frame index increment is applied to the reference stream.
    \item \textbf{Dropped frames:} When $\Delta t \approx k \cdot (1/f_m)$ for integer $k \geq 1$, the number of missing frames $k$ is compensated via $k = \text{round}(\Delta t \cdot f_m) - 1$.
\end{itemize}

This compensation is applied independently when aligning the RealSense depth stream with the CVS stream, ensuring that depth frames are matched to their correct temporal positions.

\paragraph{Color-Depth Phase Calibration via Frame Counter Matching}
To resolve the intrinsic phase offset between the RealSense color and depth sensors (which can exhibit $\geq$4\,ms timestamp discrepancies for frames with identical hardware Frame Counters), we implement a precise phase calibration based on hardware frame counters.

Specifically, we first extract the common Frame Counters $\mathcal{C}_{hw}$ that appear in both the color and depth data streams. For each common counter $c$, we compute the timestamp difference:
\begin{equation}
    \Delta t_{cd}(c) = (t_{c}^{\text{Color}} - t_{c}^{\text{Depth}}) / 1000
\end{equation}
and the corresponding phase offset:
\begin{equation}
    \phi(c) = \text{round}\left(\frac{\Delta t_{cd}(c)}{1/f_m}\right)
\end{equation}
A depth-to-color mapping $\mathcal{M}: \mathbb{N} \to \mathbb{N}$ is then constructed, and frames without a shared Frame Counter (indicating missed hardware synchronization) are discarded. After this calibration, only frames that are simultaneously aligned across all three streams are retained for final output.

\subsection{Camera Calibration}
Accurate geometric calibration is a prerequisite for the projective transformation described below. We perform a three-stage calibration procedure using a printed planar chessboard target ($11 \times 8$ corners, $29.66$\,mm square size).

\paragraph{Intrinsic Calibration}
For each camera independently, we collect multiple observations of the chessboard at varying poses and distances. Corner detection is performed at $2\times$ upscaled resolution followed by \texttt{cornerSubPix} sub-pixel refinement, yielding stable sub-pixel accuracy. An iterative outlier rejection loop discards frames with reprojection error exceeding a defined threshold, progressively refining the intrinsic parameters. For the CVS (native resolution $640 \times 320$), the calibrated intrinsic matrix $\mathbf{K}_t$ is stored in \texttt{intrinsic\_tianmou.json}. For the RealSense D455, to avoid stretching distortions during spatial alignment, we crop the native $640 \times 480$ depth maps to $640 \times 320$ and perform a dedicated intrinsic recalibration on the cropped imagery, producing $\mathbf{K}_s$ stored in \texttt{intrinsic\_realsense\_cropped.json}. A note on the CVS: due to the physical mounting orientation, all CVS images are horizontally flipped (\texttt{cv2.flip(I, 1)}) prior to both calibration and downstream use.

\paragraph{Joint Stereo Extrinsic Calibration}
With both intrinsic matrices established, we perform a joint stereo calibration between the CVS and RealSense cameras using \texttt{cv2.stereoCalibrate}. The resulting extrinsic parameters satisfy:
\begin{equation}
    \mathbf{P}_{tm} = \mathbf{R} \cdot \mathbf{P}_{rs} + \mathbf{T}
\end{equation}
where $\mathbf{P}_{cvs}$ and $\mathbf{P}_{rs}$ are 3D points in the CVS and RealSense camera coordinate systems, respectively. Our final calibration achieves a mean reprojection error of $\sim$0.2 pixels and a mean epipolar error of $\sim$0.10 pixels.

\paragraph{Calibration Validation}
The quality of the extrinsic calibration is validated through three metrics: (1) \textbf{Epipolar distance error}: for each corner detected in the CVS image, its corresponding epipolar line in the RealSense image is measured; (2) \textbf{Triangulation reprojection error}: 3D points are reconstructed via triangulation and re-projected into both views; (3) \textbf{Physical layout verification}: the sign and magnitude of the translation vector $\mathbf{T}$ are checked to confirm the physical positioning.

\subsection{Spatial Alignment and Projective Geometry}
To construct pixel-aligned ground-truth depth maps for the CVS, the measurements from the RealSense must be accurately reprojected.

\paragraph{Distortion-Free FOV Matching}
To avoid stretching distortions, we adopt a \textbf{Symmetric Vertical Cropping} strategy on the RealSense images, removing 80-pixel margins from the top and bottom to match the $640 \times 320$ resolution. We perform a complete intrinsic recalibration on the cropped imagery using a chessboard target.

\paragraph{Cross-Modal Reprojection}
Using the calibrated parameters, raw depth maps are geometrically transformed into the CVS's coordinate system. For a source pixel $\mathbf{u}_s = [u_s, v_s, 1]^T$, its target projection $\mathbf{u}_t$ is:
\begin{equation}
    z_t \mathbf{u}_t = \mathbf{K}_t \left( \mathbf{R} \mathbf{K}_s^{-1} \mathbf{u}_s z_s + \mathbf{T} \right)
\end{equation}
where $\mathbf{R} \in \text{SO}(3)$ and $\mathbf{T} \in \mathbb{R}^3$ are obtained via joint extrinsic calibration, achieving an average reprojection error of $\sim$0.2 pixels.

\paragraph{Refinement and Artifact Rejection}
We employ a \textbf{Z-buffer competition} mechanism to resolve occlusion conflicts by retaining only the minimum depth value $z_t^* = \min \{ z_{t,i} \}$. Finally, to suppress "flying pixels" near object boundaries, we apply a \textbf{local median outlier detection} filter with a $7 \times 7$ sliding window. Pixels whose depth exceeds the local median by more than $\tau = 350$\,mm are removed, targeting background spray while preserving foreground details.

\subsection{Triple-Stream Aligned Dataset Export}

\paragraph{Index-Driven Frame Filtering}
The post-processing pipeline generates a JSON index file. During export, only frames where both \texttt{is\_valid = True} and \texttt{has\_color = True} are retained, ensuring that every exported sample contains valid data from all three streams ($\mathcal{C}$, RealSense Color, and $\mathcal{D}_{gt}$).

\paragraph{Frame-Level Stream Matching}
For each exported frame:
\begin{itemize}
    \item \textbf{CVS RGB} is read at the indexed \texttt{cvs\_idx}.
    \item \textbf{RealSense Color and Depth} are matched using the hardware \texttt{Frame Counter} stored in metadata files (e.g., \url{frame_Color_metadata_1774259671696.23559570312500.txt}).
    \item \textbf{Raw 16-bit depth} (Z16 format) is extracted to preserve precision.
\end{itemize}

\paragraph{Dataset Organization}
The export produces four subdirectories: \texttt{cvs/} (aligned RGB), \texttt{color/} (RealSense Color), \texttt{depth/} (raw Z16 depth), and \texttt{comparison/} (three-panel visualization with unified info bars). 

\end{document}

-------------------------------------------------------
需要检查的代码区域分四个步骤，也就是环节
0、传感器初始化
yyf/realsense.cpp
yyf/main.cpp
1、拍摄
yyf/cxr_files/tianmou_auto_gui.py
2、后处理-三路对齐
yyf/post_process.py
3、后处理-三路输出
yyf/export_aligned_frames.py
4、标定
yyf/calib   此目录下代码不全是有用的，有些没用
5、深度投影（包含了所选的标定数据，内参外参）
yyf/calib/depth_rec.py