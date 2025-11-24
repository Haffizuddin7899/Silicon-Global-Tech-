# Machine Learning-Based CPU Burst Time Prediction for Process Scheduling: A Review and Comparative Analysis

**Author:** Haffizuddin7899  
**Institution:** [Your Institution Name]  
**Course:** Operating Systems  
**Date:** November 23, 2025  
**Document Type:** Review & Comparative Analysis Paper

---

## ABSTRACT

CPU scheduling is a fundamental aspect of operating system performance optimization. Traditional scheduling algorithms such as Shortest Job First (SJF) and Shortest Remaining Time First (SRTF) theoretically minimize average waiting and turnaround times but require prior knowledge of CPU burst times—information typically unavailable at runtime. This review paper analyzes three recent research works that employ Machine Learning (ML) techniques to predict CPU burst times, thereby enabling practical implementation of optimal scheduling algorithms. The first paper by Samal et al. (2022) [1] compares ensemble ML methods including Random Forests, XGBoost, K-Nearest Neighbors, and Decision Trees on the GWA-T-1 DAS2 dataset, achieving superior performance with Random Forests (MAE = 14.150 ms). The second paper by Effah et al. (2025) [2] implements six ML models including Artificial Neural Networks and Support Vector Machines on synthetic data mimicking GWA-T-4, with ANN achieving the best accuracy (MAE ≈ 4.13 ms). The third paper by Moni et al. (2022) [3] proposes a novel Absolute Difference Based Time Quantum Round Robin (ADRR) algorithm combined with ML prediction, demonstrating approximately 2x performance improvement over traditional algorithms. Through comparative analysis, this review identifies research gaps, evaluates methodological approaches, and discusses implications for intelligent, adaptive operating systems. The findings support the development of ML-integrated scheduling systems that can significantly reduce waiting times, turnaround times, and context switching overhead in modern computing environments.

**Keywords:** CPU Scheduling, Machine Learning, Burst Time Prediction, Process Scheduling, SJF, SRTF, Round Robin, Operating Systems

---

## I. INTRODUCTION

### A. Background and Context

Central Processing Unit (CPU) scheduling is one of the most critical components of operating system design, directly impacting system performance, resource utilization, and user experience [4]. In multiprogramming and multitasking environments, the CPU scheduler determines which process receives processor time and for how long, affecting metrics such as throughput, waiting time, turnaround time, and response time [5]. The efficiency of scheduling algorithms becomes increasingly important in modern computing environments characterized by diverse workloads, heterogeneous hardware, cloud computing infrastructures, and real-time systems [6].

Traditional CPU scheduling algorithms have been developed and refined over decades. First-Come-First-Served (FCFS) is simple but suffers from the convoy effect [7]. Round Robin (RR) provides fairness through time quantum allocation but may increase context switching overhead [8]. Priority-based scheduling can lead to starvation of low-priority processes [9]. Among these algorithms, Shortest Job First (SJF) and its preemptive variant, Shortest Remaining Time First (SRTF), are theoretically optimal for minimizing average waiting time [10].

### B. The Fundamental Problem

Despite their theoretical optimality, SJF and SRTF face a significant practical limitation: they require accurate prior knowledge of each process's CPU burst time—the duration of continuous CPU execution before the process blocks for I/O or terminates [11]. In real-world systems, this information is typically unavailable when scheduling decisions must be made. Traditional approaches to estimating burst times rely on Exponential Averaging (EA), a mathematical heuristic that calculates the predicted next burst as:

τ_{n+1} = α·t_n + (1-α)·τ_n

where τ_{n+1} is the predicted burst time, t_n is the actual length of the nth CPU burst, α is a smoothing factor (0 ≤ α ≤ 1), and τ_n is the previous prediction [12]. While computationally simple, EA lacks adaptability to dynamic workloads and often produces unreliable estimates, particularly in heterogeneous computing environments with varying process behaviors [1].

### C. Machine Learning as a Solution

Recent advances in Machine Learning offer promising alternatives to traditional prediction methods. ML algorithms can learn complex, non-linear patterns from historical execution data, potentially providing more accurate and adaptive burst time predictions [13]. Supervised learning techniques including Decision Trees, Random Forests, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Gradient Boosting methods like XGBoost, and Artificial Neural Networks (ANN) have been applied to this problem with varying degrees of success [14].

The integration of ML into CPU scheduling represents a paradigm shift from static, rule-based approaches to intelligent, data-driven decision-making [15]. By training models on historical process attributes—such as previous burst times, memory usage, I/O wait times, process priorities, and resource requirements—ML systems can potentially predict future burst times with higher accuracy than conventional methods [16]. This capability enables practical deployment of theoretically optimal algorithms like SJF and SRTF in real-world operating systems.

### D. Importance and Motivation

Accurate CPU burst time prediction is crucial for several reasons:

1. **Performance Optimization:** Better predictions lead to more efficient scheduling decisions, reducing average waiting and turnaround times, thereby improving overall system responsiveness [17].

2. **Resource Utilization:** Accurate burst time estimates enable better CPU allocation, maximizing utilization while minimizing idle time [18].

3. **Starvation Prevention:** Improved scheduling fairness helps prevent process starvation, where long processes are indefinitely postponed [19].

4. **Context Switching Reduction:** Better time quantum calculations in RR algorithms reduce unnecessary context switches, which incur computational overhead [20].

5. **Energy Efficiency:** Optimized scheduling can reduce energy consumption in data centers and mobile devices—critical concerns in modern computing [21].

6. **Heterogeneous Computing:** As computing infrastructures become increasingly heterogeneous (cloud, grid, high-performance computing), adaptive prediction becomes essential for managing diverse workloads across different hardware configurations [22].

### E. Scope and Objectives

This review paper analyzes three recent research contributions that apply ML techniques to CPU burst time prediction for process scheduling optimization:

1. **Samal, Jha, and Goyal (2022) [1]:** "CPU Burst-Time Estimation using Machine Learning" - Comparative study of ensemble ML techniques on grid computing datasets.

2. **Effah et al. (2025) [2]:** "Predicting CPU Burst Times with ML to Enhance Shortest Job First (SJF) and Shortest Remaining Time First (SRTF) CPU Scheduling" - Comprehensive evaluation of six ML models with focus on SJF/SRTF integration.

3. **Moni et al. (2022) [3]:** "Comparative Analysis of Process Scheduling Algorithm using AI models" - Novel scheduling algorithm proposal with ML-based time quantum optimization.

The objectives of this review are to:
- Summarize the research problems, methodologies, and key contributions of each paper
- Compare and contrast the approaches, datasets, and experimental results
- Identify research gaps and limitations in current methodologies
- Discuss implications for future research and practical implementations
- Provide insights to support subsequent implementation work in this domain

### F. Organization

The remainder of this paper is organized as follows: Section II provides detailed summaries of each of the three papers, including their research problems, methodologies, key contributions, and limitations. Section III presents a comprehensive comparative analysis through tabular comparison and critical discussion. Section IV analyzes research gaps, innovative methods, and differences in approaches and results. Section V concludes with major findings and recommendations for future work, explaining how this review supports subsequent implementation assignments.

---

## II. SUMMARY OF EACH PAPER

### A. Paper 1: CPU Burst-Time Estimation using Machine Learning

**Citation:** P. Samal, S. Jha, and R. K. Goyal, "CPU Burst-Time Estimation using Machine Learning," in *2022 IEEE Delhi Section Conference (DELCON)*, New Delhi, India, 2022, pp. 1-6, doi: 10.1109/DELCON54057.2022.9753639.

#### 1) Research Problem

The authors address the fundamental challenge that traditional CPU scheduling algorithms such as SJF and Multi-level Queue scheduling require prior knowledge of CPU burst times, which is typically unavailable in real-world systems [1]. They identify that the conventional Exponential Averaging method for burst time prediction often produces unreliable and inaccurate estimates, particularly in dynamic computing environments. The research question is: Can Machine Learning techniques provide more accurate CPU burst time predictions than traditional statistical methods, thereby enabling efficient implementation of optimal scheduling algorithms?

#### 2) Methodology

**Dataset:** The study utilizes the GWA-T-1 DAS2 grid workload dataset from the Grid Workloads Archive of TU-Delft [23]. The dataset contains 1,124,772 jobs with 29 attributes. The researchers randomly selected 30,000 jobs for model training and evaluation.

**Data Preprocessing:**
- Removed 13 constant-value features that did not contribute to model variance
- Eliminated 8 features deemed irrelevant to CPU burst time calculation through manual analysis
- Applied Chi-Square test for feature selection using Scikit-learn's SelectKBest class [24]
- Identified key predictive features: UsedCPUTime (score: 4268.76), UsedMemory (score: 2655.10), ReqTime (score: 1928.22), and others
- Used Leave-One-Out encoding for categorical variables to convert them to numerical format
- Final feature set: 11 attributes after feature selection

**Machine Learning Models:**
The study implemented four ML regression algorithms:

1. **K-Nearest Neighbors (KNN):** Distance-based model initially configured with k=3 neighbors, optimized to k=4 through grid search hyperparameter tuning [25].

2. **Decision Tree (DTree):** CART algorithm with least-squares impurity measure [26]. Optimized parameters: max_depth=9, min_samples_leaf=3, min_samples_split=2.

3. **XGBoost (XGB):** Gradient boosting ensemble with 200 estimators [27]. The study established a strong relationship between MAE and number of estimators through experimental analysis.

4. **Random Forest (RF):** Ensemble of 200 decision trees using bagging [28]. Optimized parameters: n_estimators=55 (approximately), criterion='mse' (mean squared error). The model minimizes Residual Sum of Squares (RSS) at each node split.

**Train-Test Split:** 70% training data, 30% testing data (21,000 training samples, 9,000 test samples)

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)  
- R-Squared Score (R²)

#### 3) Key Contributions

**Experimental Results:**

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Random Forest | **14.150** | **291.774** | **0.992** |
| XGBoost | 36.141 | 342.430 | 0.990 |
| Decision Tree | 21.652 | 815.458 | 0.943 |
| KNN | 242.703 | 2215.790 | 0.582 |

**After Hyperparameter Optimization:**

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Decision Tree (optimized) | 31.182 | 570.391 | 0.972 |
| KNN (optimized) | 203.899 | 1964.153 | 0.671 |

**Key Findings:**
1. Random Forest achieved the best performance across all metrics (MAE=14.150, R²=0.992)
2. Strong correlation identified between CPU burst time and UsedCPUTime and UsedMemory features
3. Tree-based models (RF, XGBoost, DTree) significantly outperformed KNN
4. Feature selection using Chi-Square test reduced dimensionality while maintaining prediction accuracy
5. ML models substantially outperformed traditional Exponential Averaging

#### 4) Strengths

1. **Comprehensive Model Comparison:** Evaluated multiple ML algorithms with rigorous hyperparameter optimization
2. **Robust Feature Engineering:** Applied systematic feature selection methodology (Chi-Square test) to identify most predictive attributes
3. **Real-World Dataset:** Used authentic grid computing workload data (GWA-T-1 DAS2) with over 1 million jobs
4. **Clear Methodology:** Well-documented preprocessing steps, model configurations, and evaluation procedures
5. **Practical Applicability:** Demonstrated feasibility of ML integration into scheduling algorithms
6. **Statistical Rigor:** Used multiple evaluation metrics (MAE, RMSE, R²) providing comprehensive performance assessment

#### 5) Limitations

1. **Limited Dataset Scope:** Only 30,000 of 1.1+ million available jobs were used; larger sample might improve generalization
2. **Single Dataset:** Validation limited to one grid computing dataset; generalizability to other workload types uncertain
3. **No Real-Time Testing:** Performance evaluated offline; real-time scheduling integration not demonstrated
4. **Cold Start Problem:** Not addressed how the system handles new processes without historical data
5. **Computational Overhead:** Training and prediction time costs not analyzed; critical for real-time systems
6. **Support Vector Machines:** Briefly mentioned but not thoroughly evaluated; could provide additional comparison baseline
7. **Scheduling Integration:** Predicted burst times not integrated into actual scheduling algorithm implementations
8. **Class Imbalance:** Did not discuss whether dataset exhibits burst time distribution imbalances that might affect model performance

---

### B. Paper 2: Predicting CPU Burst Times with ML to Enhance SJF and SRTF CPU Scheduling

**Citation:** E. Effah, S. J. Atsu, Z. A. Brew, J. K. Mensah, E. A. Quaicoe, M. O. Ansah, A. Yeful, and R. P. Baffoe, "Predicting CPU Burst Times with ML to Enhance Shortest Job First (SJF) and Shortest Remaining Time First (SRTF) CPU Scheduling," *International Journal of Computer Science and Information Security (IJCSIS)*, vol. 23, no. 5, pp. 1-10, September 2025, doi: 10.5281/zenodo.17131785.

#### 1) Research Problem

The authors identify that while SJF and SRTF are theoretically optimal for reducing average waiting and turnaround times, their practical deployment is severely hindered by the impossibility of knowing CPU burst times beforehand [2]. Traditional Exponential Averaging methods lack adaptability to dynamic and heterogeneous workloads [29]. The research addresses: How can Machine Learning be leveraged to accurately predict CPU burst times, enabling practical real-time deployment of SJF and SRTF in modern computing environments? The study also emphasizes the importance of resource optimization for environmental sustainability and energy efficiency in data centers [21].

#### 2) Methodology

**Dataset:** Initially used a synthetic dataset with 1,000 samples designed to mimic the structure of the Grid Workload Archive GWA-T-4 dataset [30]. The synthetic data was generated to simulate realistic grid workload patterns with plans to transition to real GWA-T-4 data.

**Features (7 attributes):**
- Process arrival time (normalized 0-100 ms)
- Previous CPU burst length (0-50 ms)
- I/O wait time (0-20 ms)
- Priority level (1-5)
- Memory consumption (100-2000 MB)
- Number of I/O requests (0-10)
- **Target variable:** CPU burst time (0-50 ms), generated as: 0.7×prev_cpu_burst + 0.3×io_wait_time + noise

**Data Preprocessing:**
- Removed missing values
- Standardized features using StandardScaler for uniform scaling [31]
- Engineered additional feature: io_to_burst_ratio
- Applied Correlation-Based Feature Selection (CFS) and Recursive Feature Elimination (RFE) [32]
- Final selected features: io_to_burst_ratio, prev_cpu_burst, io_wait_time

**Machine Learning Models (6 algorithms):**

1. **K-Nearest Neighbors (KNN):** k=7 neighbors, uniform weights, Euclidean distance metric [25]

2. **Support Vector Machines (SVM):** Linear kernel, regularization parameter C=10, epsilon=0.1 for regression [33]

3. **Decision Trees:** Hierarchical decision rules, max_depth=5, min_samples_split=2 [26]

4. **Random Forest:** Ensemble of 200 trees, max_depth=10 [28]

5. **XGBoost:** Gradient boosting with 200 estimators, learning_rate=0.01 [27]

6. **Artificial Neural Networks (ANN):** Multi-layer perceptron with single hidden layer of 100 nodes, tanh activation function, L2 regularization (α=0.001) [34]

Additionally, an **Ensemble Model** combining ANN + SVM was evaluated.

**Hyperparameter Optimization:** GridSearchCV with 5-fold cross-validation [35]

**Train-Test Split:** 80% training, 20% testing with stratified sampling

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Relative Absolute Error (RAE)
- Correlation Coefficient (CC)

**Baseline Comparison:** Exponential Averaging (traditional method)

**Scheduling Integration:**
1. Extract and standardize features from processes in ready queue
2. Predict CPU burst time using trained ML model
3. Sort processes by predicted burst time (SJF) or remaining time (SRTF)
4. Execute Python-based simulation engine
5. Collect performance metrics

#### 3) Key Contributions

**Prediction Performance Results:**

| Model | MAE (ms) | RMSE (ms) | RAE | CC |
|-------|----------|-----------|-----|-----|
| **ANN** | **4.1338** | **5.2766** | **0.4395** | **0.8853** |
| **Ensemble (ANN+SVM)** | **4.1182** | **5.2626** | **0.4379** | **0.8855** |
| SVM | 4.1927 | 5.3056 | 0.4458 | 0.8832 |
| XGBoost | 4.4123 | 5.7044 | 0.4691 | 0.8600 |
| Random Forest | 4.6717 | 5.9763 | 0.4967 | 0.8561 |
| K-Nearest Neighbors | 4.7852 | 5.9957 | 0.5088 | 0.8557 |
| Decision Tree | 4.8070 | 6.0738 | 0.5111 | 0.8527 |
| **Baseline (EA)** | **20.6733** | **23.2376** | **2.1981** | **0.8728** |

**Scheduling Performance Results:**

| Scheduling Strategy | Avg Waiting Time (ms) | Avg Turnaround Time (ms) |
|---------------------|----------------------|-------------------------|
| SJF + ANN | 1483.11 | 1503.72 |
| SRTF + ANN | 1483.05 | 1503.67 |
| SJF + Ensemble | 1483.36 | 1503.97 |
| SRTF + Ensemble | 1483.30 | 1503.92 |
| Baseline (EA) | 1586.37 | 1606.98 |
| Oracle (Perfect) | 1411.23 | 1432.89 |

**Key Findings:**
1. ANN and ANN+SVM Ensemble achieved superior prediction accuracy (MAE ≈ 4.12-4.13 ms)
2. ML models reduced MAE by approximately 80% compared to Exponential Averaging baseline
3. ML-based scheduling achieved ~6.5% improvement over baseline in waiting/turnaround times
4. ML approaches performed within 5% of theoretical "Oracle" performance (perfect predictions)
5. SRTF with ML predictions slightly outperformed SJF
6. RAE < 1.0 for all ML models indicates better-than-baseline performance

#### 4) Strengths

1. **Comprehensive Model Coverage:** Evaluated six diverse ML algorithms plus ensemble approach
2. **End-to-End Integration:** Demonstrated complete workflow from prediction to scheduling simulation
3. **Baseline Comparison:** Rigorous comparison against both traditional EA and theoretical Oracle
4. **Multiple Evaluation Metrics:** Used four complementary metrics (MAE, RMSE, RAE, CC) for thorough assessment
5. **Feature Engineering:** Applied sophisticated feature selection (CFS + RFE) and created derived features
6. **Ensemble Method:** Explored hybrid models (ANN+SVM) achieving marginal improvements
7. **Practical Focus:** Emphasized real-world deployment challenges and environmental/energy considerations
8. **Mathematical Rigor:** Provided detailed mathematical formulations for all ML algorithms
9. **Statistical Validation:** 5-fold cross-validation with stratified sampling ensures robust results

#### 5) Limitations

1. **Synthetic Data:** Primary experiments used simulated data rather than real-world workloads
2. **Small Dataset:** Only 1,000 samples; may not capture complexity of production systems
3. **High Absolute Times:** Simulated waiting/turnaround times (≈1483-1503 ms) far exceed target range (0-50 ms), attributed to simulation artifacts
4. **Simulation Limitations:** Authors acknowledge simulation engine issues causing inflated timing metrics
5. **No Real OS Implementation:** Scheduling integration tested only in simulation, not actual operating system
6. **Cold Start Not Addressed:** No discussion of handling new processes without historical data
7. **Computational Cost:** Training and prediction overhead not quantified
8. **Limited Workload Diversity:** Single dataset type; generalization to batch processing, interactive tasks unclear
9. **Hyperparameter Search Space:** Grid search ranges not fully documented
10. **Scalability:** Not tested on larger datasets (e.g., 100,000+ processes)

---

### C. Paper 3: Comparative Analysis of Process Scheduling Algorithm using AI Models

**Citation:** Md. M. A. Moni, Maharshi Niloy, A. H. Chowdhury, F. J. Khan, Md. F. Juboraj, and A. Chakrabarty, "Comparative Analysis of Process Scheduling Algorithm using AI models," in *2022 25th International Conference on Computer and Information Technology (ICCIT)*, Cox's Bazar, Bangladesh, 2022, pp. 1-6, doi: 10.1109/ICCIT57492.2022.10055395.

#### 1) Research Problem

The authors identify multiple interconnected problems in CPU scheduling [3]:
1. Most current algorithms rely on "educated assumptions" rather than data-driven decisions
2. Process behavior varies unpredictably across different hardware (cloud, grid, HPC) and operating systems (Mac, Windows, Linux)
3. Traditional approaches lead to problems: process starvation, excessive context switching, deadlocks, inefficient CPU utilization, memory wastage [36]
4. Lack of analytical data on scheduling algorithm performance and optimal time quantum calculation
5. Operating systems lack intelligence to make adaptive scheduling decisions

The central research question: Can a novel scheduling algorithm combined with ML-based burst time prediction achieve better performance than existing Round Robin variants in terms of average waiting time, turnaround time, and context switches?

#### 2) Methodology

**Dataset:** GWA-T-4 AuverGrid dataset from Grid Workloads Archive (TU-Delft) [37]
- Production grid platform with 5 geographically distributed clusters in Auvergne, France
- 475 CPUs, 405 users, dual 3GHz Pentium-IV Xeon nodes
- **Total jobs:** 404,176 with 29 attributes
- **Used for training/testing:** 347,611 rows after preprocessing

**Data Preprocessing (Extensive):**
1. Dropped 11 columns containing null values (represented as -1 in dataset)
2. Removed 'ReqNProcs' (constant single value across dataset)
3. Eliminated rows with null values in 'ReqMemory' column
4. Encoded categorical string variables to numeric:
   - QueueID → QueueNo
   - GroupID → GroupNo
   - ExecutableID → ExecutableNo
   - OrigSiteID → OrigSiteNo
   - LastRunSiteID → LastRunSiteNo (later dropped due to full correlation with OrigSiteNo)
   - UserID → UserNo
5. Dropped JobID (fully correlated with SubmitTime)
6. Applied feature importance scoring using Linear Regression [38]
7. Kept only features with importance score > 0.1
8. **Final dimensions:** 347,611 × 9 (8 features + 1 target 'RunTime')

**Top Features by Importance Score:**
- AverageCPUTimeUsed (73.58)
- UsedMemory (14.05)
- ReqTime (2.57)
- SubmitTime (1.31)
- WaitTime (0.96)
- PartitionID (0.55)
- OrigSiteNo (0.53)
- Status (0.87)

**Machine Learning Models (4 algorithms):**

1. **Linear Regression (LR):** Supervised learning finding linear correlation between features and burst time [38]

2. **K-Nearest Neighbors (KNN):** Non-parametric classifier using Euclidean distance, optimized number of neighbors [25]

3. **Decision Tree (DT):** CART algorithm with entropy and information gain for node splitting [26]

4. **Multi-Layer Perceptron (MLP):** Neural network with 20 hidden layers, learning rate = 0.001 [39]

**Additional Baseline:** Exponential Averaging (26.08% accuracy)

**Train-Test Split:** 70% training, 30% testing

**Proposed Scheduling Algorithm: Absolute Difference Based Time Quantum Round Robin (ADRR)**

**ADRR Algorithm Logic:**
1. Add incoming process p with process_id to process_list
2. Sort processes by ascending remaining burst time
3. Add priority processes to front of process_list
4. **Calculate dynamic time quantum:**
   ```
   time_quantum = abs(process_list.values()[-1] - process_list.values()[0])
   
   if time_quantum == 0 and first_round:
       time_quantum = process_list.values()[-1]
   
   if time_quantum == 0:
       time_quantum = time_quantum_previous_round
   ```
5. Execute process according to time quantum
6. Repeat until process_list is empty

**Comparison Algorithms:**
1. Traditional Round Robin (RR) [40]
2. Modified Round Robin Algorithm (MRRA) [41]
3. Self-Adjustment Round Robin (SARR) [42]
4. Optimized Round Robin (ORR) [43]

**Evaluation Metrics:**
- Average Turnaround Time (avg TT)
- Average Waiting Time (avg WT)
- Number of Context Switches (CS)

**Experimental Setup:**
- Primary test: 10,000 processes
- Consecutive test: 10 epochs with 500-5000 processes (increments of 500)
- Platform: Jupyter Notebook, Python 3
- Hardware: AMD Ryzen 7 5800X, 64GB RAM, Nvidia RTX 3060, Windows 10

#### 3) Key Contributions

**ML Model Prediction Accuracy:**

| Model | Accuracy Score |
|-------|---------------|
| **Decision Tree (DT)** | **98.64%** |
| Linear Regression (LR) | 97.96% |
| Neural Network (MLP) | 26.01% |
| K-Nearest Neighbors (KNN) | 17.1% |
| Exponential Average | 26.08% |

**Scheduling Performance (10,000 processes):**

| Algorithm | Avg TT (ms) | Avg WT (ms) | Context Switches |
|-----------|-------------|-------------|-----------------|
| **ADRR (Proposed)** | **40,331,930.48** | **40,312,117.96** | **20,002** |
| Traditional RR | 87,194,390.98 | 87,174,578.46 | 28,964 |
| SARR | 72,398,064.70 | 72,378,252.18 | 39,956 |
| MRRA | 84,924,105.36 | 84,904,292.84 | 25,208 |
| ORR | 78,508,779.73 | 78,488,967.20 | 22,470 |

**Performance Improvement:**
- **~54% reduction** in average turnaround time vs. Traditional RR
- **~46% reduction** in average waiting time vs. Traditional RR
- **~31% reduction** in context switches vs. Traditional RR
- Consistently approximately **2x faster** than other algorithms

**Consecutive Test Results (500-5000 processes):**
At 5000 processes:
- ADRR: avg TT = 962,821.14, avg WT = 961,586.58, CS = 10,000
- SARR: avg TT = 1,398,027.61, avg WT = 1,396,793.04, CS = 19,694
- MRRA: avg TT = 1,440,608.90, avg WT = 1,439,374.33, CS = 15,162
- ORR: avg TT = 1,803,293.73, avg WT = 1,802,059.17, CS = 120,080
- RR: avg TT = 2,327,200.60, avg WT = 2,324,731.47, CS = 22,858

**Key Findings:**
1. Decision Tree achieved highest burst time prediction accuracy (98.64%)
2. ADRR algorithm consistently outperformed all comparison algorithms across all process loads
3. Dynamic time quantum calculation based on absolute difference reduced overhead
4. Sorting mechanism time complexity made negligible by high algorithm performance
5. ML-based prediction enables intelligent, adaptive scheduling decisions
6. Performance advantage maintained across small (500) to large (5000+) process batches

#### 4) Strengths

1. **Largest Real Dataset:** Used 347,611 real grid computing jobs—significantly larger than other studies
2. **Novel Algorithm:** Proposed original ADRR with dynamic time quantum calculation
3. **Extensive Preprocessing:** Thorough 9-step data cleaning and feature engineering process
4. **Comprehensive Comparison:** Evaluated against 4 existing algorithms across multiple metrics
5. **Scalability Testing:** Consecutive tests from 500 to 5,000 processes demonstrated consistent performance
6. **High Prediction Accuracy:** Decision Tree achieved 98.64% accuracy
7. **Practical Implementation:** Detailed pseudo-code and algorithm architecture provided
8. **Large-Scale Testing:** 10,000 process test simulates realistic high-load scenarios
9. **Consistent Performance:** ADRR maintained ~2x advantage across all test conditions
10. **Feature Importance Analysis:** Identified AverageCPUTimeUsed as strongest predictor (73.58 score)

#### 5) Limitations

1. **Extremely High Absolute Times:** Turnaround/waiting times in millions of milliseconds (e.g., 40M ms ≈ 11 hours) unrealistic for actual systems
2. **Simulation Artifacts:** High timing values suggest simulation model issues rather than real-world applicability
3. **No Real OS Integration:** Tested only in simulation environment, not kernel-level implementation
4. **MLP Poor Performance:** Neural network with 20 layers achieved only 26.01% accuracy—potential architecture/hyperparameter issues
5. **Limited MLP Exploration:** Insufficient explanation of why MLP performed so poorly
6. **Sorting Overhead:** Claimed negligible but not quantified; O(n log n) complexity could impact real-time systems
7. **Cold Start Problem:** Not addressed for new processes without historical data
8. **Priority Process Handling:** ADRR algorithm mentions priority processes but implementation details unclear
9. **Energy Consumption:** Not measured despite being critical metric for modern systems
10. **Context Switch Quality:** Only counted switches, didn't analyze overhead cost per switch

---

## III. COMPARATIVE ANALYSIS

### A. Comparative Table

The following table provides a comprehensive comparison of the three papers across multiple dimensions:

| **Aspect** | **Paper 1: Samal et al. (2022)** | **Paper 2: Effah et al. (2025)** | **Paper 3: Moni et al. (2022)** |
|------------|----------------------------------|----------------------------------|---------------------------------|
| **Problem Addressed** | SJF/SRTF require burst time knowledge; EA unreliable | SJF/SRTF impractical due to unknown burst times; EA lacks adaptability | OS lacks intelligence for scheduling; educated assumptions insufficient; multiple system inefficiencies |
| **Dataset** | GWA-T-1 DAS2 (1.1M jobs, used 30K) | Synthetic mimicking GWA-T-4 (1K samples) | GWA-T-4 AuverGrid (404K jobs, used 347K) |
| **Dataset Size** | 30,000 samples | 1,000 samples | 347,611 samples |
| **Features** | 11 after selection from 29 | 3 after selection from 7 | 8 after selection from 29 |
| **ML Methods Used** | KNN, Decision Tree, XGBoost, Random Forest | KNN, SVM, Decision Tree, Random Forest, XGBoost, ANN, Ensemble (ANN+SVM) | KNN, Linear Regression, Decision Tree, MLP (20 layers) |
| **Best Model** | Random Forest | ANN & ANN+SVM Ensemble | Decision Tree |
| **Best MAE** | 14.150 ms (RF) | 4.1182 ms (Ensemble) | Not reported (98.64% accuracy) |
| **Best RMSE** | 291.774 ms (RF) | 5.2626 ms (Ensemble) | Not reported |
| **Best R²/Accuracy** | 0.992 (RF) | CC=0.8855 (Ensemble) | 98.64% (DT) |
| **Baseline Comparison** | Implied EA comparison | EA: MAE=20.67 ms | EA: 26.08% accuracy |
| **Train-Test Split** | 70-30% | 80-20% stratified | 70-30% |
| **Feature Selection** | Chi-Square (SelectKBest) | CFS + RFE | Feature importance score (LR) |
| **Hyperparameter Tuning** | Grid Search | GridSearchCV 5-fold CV | Not explicitly mentioned |
| **Scheduling Integration** | Theoretical discussion only | Python simulation (SJF/SRTF) | Novel ADRR algorithm implementation |
| **Scheduling Algorithms** | SJF, SRTF mentioned | SJF, SRTF simulated | ADRR vs. RR, SARR, MRRA, ORR |
| **Avg Waiting Time** | Not measured | 1483.05 ms (SRTF+ANN) | 40,312,117.96 ms (ADRR, 10K processes) |
| **Avg Turnaround Time** | Not measured | 1503.67 ms (SRTF+ANN) | 40,331,930.48 ms (ADRR, 10K processes) |
| **Context Switches** | Not measured | Not measured | 20,002 (ADRR, 10K processes) |
| **Performance Improvement** | Not quantified | ~6.5% over EA baseline; within 5% of Oracle | ~54% over RR; 2x faster than alternatives |
| **Evaluation Metrics** | MAE, RMSE, R² | MAE, RMSE, RAE, CC | Accuracy, avg TT, avg WT, CS |
| **Novel Contribution** | Comparative study of ensemble ML on grid data | Comprehensive ML+scheduling integration | Novel ADRR algorithm with ML prediction |
| **Computational Platform** | Not specified | Intel i7, 8GB RAM, Windows 10 | AMD Ryzen 7 5800X, 64GB RAM, RTX 3060 |
| **Implementation Tool** | Python, Scikit-learn | Python, Scikit-learn, Pandas, NumPy | Jupyter Notebook, Python 3 |
| **Real OS Testing** | No | No | No |
| **Scalability Testing** | Single 30K sample | Single 1K sample | Consecutive tests: 500-5000 processes |
| **Key Strength** | Rigorous model comparison with real data | End-to-end prediction-to-scheduling pipeline | Largest dataset, novel algorithm, extensive testing |
| **Key Limitation** | No scheduling integration | Synthetic data, simulation issues | Unrealistic absolute timing values |
| **Year Published** | 2022 | 2025 | 2022 |
| **Publication Venue** | IEEE DELCON Conference | IJCSIS Journal | IEEE ICCIT Conference |

### B. Dataset Comparison

**Dataset Characteristics:**

1. **Paper 1 (GWA-T-1 DAS2) [23]:**
   - **Type:** Real grid computing workload
   - **Origin:** Advanced School for Computing and Imaging (ASCI), Netherlands
   - **Scale:** 1,124,772 jobs (used 30,000)
   - **Attributes:** 29 features
   - **Utilization:** Random sampling (2.7% of total)
   - **Advantage:** Real-world production data
   - **Limitation:** Small sample relative to available data

2. **Paper 2 (Synthetic mimicking GWA-T-4) [30]:**
   - **Type:** Simulated grid workload
   - **Origin:** Synthetically generated
   - **Scale:** 1,000 samples
   - **Attributes:** 7 features
   - **Generation:** Mathematical formula with noise
   - **Advantage:** Controlled experimental conditions
   - **Limitation:** May not capture real-world complexity

3. **Paper 3 (GWA-T-4 AuverGrid) [37]:**
   - **Type:** Real grid computing workload
   - **Origin:** Production grid in Auvergne, France (5 clusters, 475 CPUs)
   - **Scale:** 404,176 jobs (used 347,611)
   - **Attributes:** 29 features → 9 after preprocessing
   - **Utilization:** 86% of total dataset
   - **Advantage:** Largest real-world dataset among three studies
   - **Limitation:** High dimensionality reduction may lose information

**Dataset Quality Assessment:**
- **Representativeness:** Papers 1 and 3 use real production data; Paper 2 uses synthetic
- **Scale:** Paper 3 > Paper 1 >> Paper 2 in terms of training samples
- **Completeness:** All papers addressed missing values; Paper 3 most thorough preprocessing
- **Diversity:** Grid workloads from different infrastructures; generalization to other domains unclear

### C. Methodology Comparison

**Machine Learning Approaches:**

| **Aspect** | **Paper 1** | **Paper 2** | **Paper 3** |
|------------|-------------|-------------|-------------|
| **Model Diversity** | 4 models | 7 models (6 + ensemble) | 4 models |
| **Ensemble Methods** | Random Forest, XGBoost | Random Forest, XGBoost, ANN+SVM | None |
| **Neural Networks** | None | ANN (MLP) | MLP (20 layers) |
| **Distance-Based** | KNN | KNN | KNN |
| **Linear Methods** | None | SVM (linear kernel) | Linear Regression |
| **Tree-Based** | DT, RF, XGBoost | DT, RF, XGBoost | DT |
| **Best Performer** | Random Forest | ANN/Ensemble | Decision Tree |

**Feature Engineering:**

| **Technique** | **Paper 1** | **Paper 2** | **Paper 3** |
|---------------|-------------|-------------|-------------|
| **Selection Method** | Chi-Square | CFS + RFE | Feature importance |
| **Derived Features** | None mentioned | io_to_burst_ratio | None mentioned |
| **Encoding** | Leave-One-Out | Not specified | Custom numeric mapping |
| **Scaling** | Not mentioned | StandardScaler | Not mentioned |
| **Final Features** | 11 | 3 | 8 |

**Evaluation Rigor:**

| **Criterion** | **Paper 1** | **Paper 2** | **Paper 3** |
|---------------|-------------|-------------|-------------|
| **Cross-Validation** | Not mentioned | 5-fold CV | Not mentioned |
| **Stratification** | No | Yes | No |
| **Multiple Metrics** | 3 (MAE, RMSE, R²) | 4 (MAE, RMSE, RAE, CC) | 4 (Accuracy, TT, WT, CS) |
| **Baseline Comparison** | Implicit | Explicit (EA + Oracle) | Explicit (EA + 4 algorithms) |
| **Hyperparameter Search** | Grid Search | GridSearchCV | Limited |

### D. Results Comparison

**Prediction Accuracy:**

Given different evaluation metrics, direct comparison is challenging. Converting to common understanding:

- **Paper 1:** R² = 0.992 indicates 99.2% variance explained (Random Forest)
- **Paper 2:** CC = 0.8855 indicates strong linear correlation; RAE = 0.4379 indicates 56% better than baseline
- **Paper 3:** 98.64% classification accuracy (Decision Tree)

**Relative Performance:**
1. All three studies demonstrated substantial improvement over traditional Exponential Averaging
2. Tree-based models performed well across all studies (RF in Papers 1 & 2, DT in Paper 3)
3. Neural networks: Excellent in Paper 2 (ANN), poor in Paper 3 (MLP)—suggests architecture sensitivity
4. KNN consistently underperformed in all three studies

**Scheduling Efficiency:**

Only Papers 2 and 3 measured scheduling performance:

- **Paper 2:** 6.5% improvement over EA baseline; within 5% of theoretical optimal
- **Paper 3:** ~54% improvement over Traditional RR; 2x faster than alternatives

Note: Absolute timing values incomparable due to different simulation scales and methodologies.

---

## IV. CRITICAL ANALYSIS

### A. Research Gaps Identified

#### 1. Real Operating System Implementation Gap

**Observation:** All three papers evaluate ML-based burst time prediction and scheduling improvements through simulations rather than actual operating system implementations.

**Implications:**
- **Kernel Integration Challenges:** None of the studies address how to integrate ML models at the kernel level [44]
- **System Call Overhead:** Prediction latency in real-time scheduling contexts not quantified
- **Memory Constraints:** In-kernel ML model memory footprint not considered
- **Interrupt Handling:** How ML prediction interacts with hardware interrupts unexplored

**Future Research Needs:**
- Kernel module development for Linux/Windows
- Real-time prediction latency benchmarking
- Lightweight model architectures suitable for kernel execution
- Comparison of user-space vs. kernel-space ML integration

#### 2. Cold Start Problem

**Observation:** None of the papers adequately address how to predict burst times for newly arrived processes without historical data [45].

**Current Approaches (Insufficient):**
- Papers assume availability of features like "previous CPU burst" which don't exist for new processes
- No discussion of bootstrapping or transfer learning approaches

**Potential Solutions Not Explored:**
- **Process Similarity Matching:** Use executable name, user ID, resource requests to find similar historical processes
- **Transfer Learning:** Pre-train models on diverse workloads, fine-tune for specific systems [46]
- **Hybrid Fallback:** Use EA or conservative estimates until sufficient process history accumulated
- **Executable Profiling:** Static code analysis to estimate computational complexity

#### 3. Workload Diversity and Generalization

**Observation:** All studies focus exclusively on grid computing workloads. Generalization to other domains unproven.

**Unexplored Workload Types:**
- **Interactive Desktop Applications:** Word processors, web browsers with user-driven behavior
- **Batch Processing:** Scientific simulations, data analytics pipelines
- **Real-Time Systems:** Embedded systems, industrial control with strict deadlines [47]
- **Database Workloads:** Transaction processing, OLAP queries
- **Mobile/Edge Computing:** Resource-constrained, power-aware environments

**Cross-Domain Challenges:**
- Feature distributions may differ significantly
- Optimal model architecture may be workload-dependent
- Grid computing assumptions (long-running jobs) don't apply to interactive tasks

#### 4. Temporal Dynamics and Concept Drift

**Observation:** None of the papers address how workload patterns change over time and how models should adapt [48].

**Unaddressed Questions:**
- How frequently should models be retrained?
- How to detect when prediction accuracy degrades due to workload shifts?
- Can online learning or incremental training be employed?
- How to balance model stability vs. adaptability?

**Potential Approaches:**
- **Sliding Window Training:** Retrain periodically on recent data
- **Concept Drift Detection:** Monitor prediction errors; retrigger training when threshold exceeded [49]
- **Ensemble Temporal Models:** Combine models trained on different time periods
- **Adaptive Learning Rates:** Increase learning when drift detected

#### 5. Energy Efficiency and Sustainability

**Observation:** Only Paper 2 mentions energy efficiency in the introduction; none measure it [21].

**Critical Omission Given:**
- Data center energy consumption represents significant operational cost and environmental impact
- Mobile/IoT devices severely power-constrained
- Green computing increasingly important in industry

**Metrics Not Measured:**
- Energy consumption per process completion
- Power-performance trade-offs
- CPU utilization vs. power draw
- Impact of prediction overhead on battery life (mobile)

#### 6. Fairness and Starvation

**Observation:** Papers focus on average-case performance; worst-case process starvation not rigorously analyzed [50].

**Concerns:**
- SJF/SRTF inherently susceptible to starvation of long processes
- Papers don't measure maximum waiting time or starvation frequency
- Fairness metrics (e.g., Jain's Fairness Index) not computed

**Unexplored Solutions:**
- Aging mechanisms to gradually increase priority of waiting processes
- Hybrid algorithms combining ML prediction with fairness guarantees
- Multi-objective optimization balancing efficiency and fairness

#### 7. Heterogeneous Computing Environments

**Observation:** All studies assume homogeneous CPU resources [51].

**Modern Computing Reality:**
- **Heterogeneous Processors:** Intel + ARM, CPU + GPU, big.LITTLE architectures
- **Virtualization:** VMs and containers with varying resource allocations
- **Cloud Elasticity:** Dynamic resource scaling
- **NUMA Architectures:** Non-uniform memory access affecting performance

**Research Needs:**
- Burst time prediction accounting for processor heterogeneity
- Resource-aware scheduling considering memory, I/O, accelerators
- Cross-platform model transferability

#### 8. Security and Privacy

**Observation:** No discussion of security implications of ML-based scheduling [52].

**Potential Vulnerabilities:**
- **Model Poisoning:** Adversaries injecting malicious training data
- **Inference Attacks:** Reverse-engineering sensitive process information from model
- **Side-Channel Leakage:** Scheduling decisions revealing confidential workload patterns
- **Resource Exhaustion:** Crafted processes exploiting prediction errors to cause DoS

**Research Directions:**
- Adversarial robustness of scheduling ML models
- Differential privacy in workload data collection
- Secure multi-party computation for distributed training

#### 9. Multi-Core and Parallel Scheduling

**Observation:** All papers assume single-processor scheduling; multi-core aspects ignored [53].

**Modern Challenges:**
- **Load Balancing:** Distributing processes across cores
- **Cache Affinity:** Preferring same core for process continuity
- **Thread-Level Parallelism:** Scheduling threads of multi-threaded applications
- **NUMA Awareness:** Minimizing cross-node memory access

#### 10. Explainability and Interpretability

**Observation:** Models treated as black boxes; no effort to explain predictions [54].

**Why It Matters:**
- **System Administrators:** Need to understand why scheduling decisions made
- **Debugging:** Identifying causes of performance anomalies
- **Trust:** Kernel-level code requires high confidence
- **Compliance:** Some domains require auditable decision-making

**Potential Approaches:**
- SHAP (SHapley Additive exPlanations) values for feature importance per prediction
- Attention mechanisms in neural networks
- Rule extraction from tree ensembles
- Contrastive explanations ("Process X scheduled because...")

### B. Innovative Methods Highlighted

#### 1. Feature Selection Techniques

**Paper 1: Chi-Square Test (SelectKBest) [24]**
- **Innovation:** Systematic statistical approach to identifying most predictive features
- **Result:** Reduced 29 features to 11, maintaining 99.2% R² score
- **Advantage:** Reduces dimensionality, mitigates overfitting, improves computational efficiency
- **Key Finding:** UsedCPUTime (score: 4268.76) and UsedMemory (score: 2655.10) dominate

**Paper 2: CFS + RFE [32]**
- **Innovation:** Dual-method feature selection combining correlation-based and recursive elimination
- **Result:** Aggressive reduction from 7 to 3 features (io_to_burst_ratio, prev_cpu_burst, io_wait_time)
- **Advantage:** Minimal feature set simplifies real-time prediction
- **Trade-off:** May sacrifice some accuracy for speed

**Paper 3: Linear Regression-Based Scoring [38]**
- **Innovation:** Training separate LR model for each feature to compute importance
- **Result:** Identified AverageCPUTimeUsed (73.58) as dominant predictor
- **Advantage:** Intuitive interpretation of feature contributions

**Comparative Insight:** Different methods converge on similar conclusions: previous CPU usage and memory consumption are strongest predictors of future burst time.

#### 2. Ensemble Approaches

**Paper 2: ANN + SVM Hybrid Ensemble [2]**
- **Innovation:** Combining neural network flexibility with SVM theoretical guarantees
- **Method:** Average predictions from independently trained ANN and SVM
- **Result:** Marginal improvement over standalone ANN (MAE: 4.1182 vs. 4.1338)
- **Insight:** Demonstrates potential of model diversity for robustness
- **Limitation:** Computational cost doubles; benefit modest

**Paper 1: Comparison of Bagging (RF) vs. Boosting (XGBoost) [1]**
- **Finding:** Random Forest outperformed XGBoost (MAE: 14.150 vs. 36.141)
- **Explanation:** Bagging's variance reduction effective for this problem; boosting may overfit
- **Insight:** Problem characteristics determine optimal ensemble strategy

#### 3. Dynamic Time Quantum Calculation

**Paper 3: Absolute Difference-Based Formula (ADRR) [3]**
- **Innovation:** Time quantum adapts each round based on process burst time distribution

**Formula:**
```
time_quantum = |max_burst - min_burst|

Special cases:
- If TQ = 0 and first_round: TQ = max_burst
- If TQ = 0: TQ = previous_TQ
```

**Advantages:**
- **Adaptive:** Automatically adjusts to workload characteristics
- **Simple:** No manual tuning required
- **Effective:** Reduces context switches by ~31% vs. traditional RR

**Comparison to Existing Approaches:**
- **SARR [42]:** Uses median burst time (static per round)
- **MRRA [41]:** Uses mean of (average burst time, highest burst time)
- **ORR [43]:** Uses max difference between adjacent sorted processes + first process burst
- **ADRR:** Most sensitive to range of burst time distribution

#### 4. Hyperparameter Optimization

**Paper 2: GridSearchCV with 5-Fold Cross-Validation [35]**
- **Innovation:** Systematic search over hyperparameter space with cross-validation
- **Robustness:** 5-fold CV ensures generalization, reduces overfitting risk
- **Example Results:**
  - KNN: k=7 (optimal from search)
  - SVM: C=10, epsilon=0.1
  - DT: max_depth=5, min_samples_split=2
  - RF: n_estimators=200, max_depth=10

**Paper 1: Iterative Grid Search [1]**
- **Innovation:** Experimental plots of MAE vs. n_estimators to visualize optimal stopping
- **Finding:** RF optimal at ~55 estimators (beyond which marginal returns diminish)
- **Insight:** Provides interpretable rationale for hyperparameter choice

#### 5. Synthetic Data Generation

**Paper 2: Controlled Simulation Approach [2]**
- **Formula:** burst_time = 0.7×prev_cpu_burst + 0.3×io_wait_time + noise
- **Rationale:** Models realistic dependency between I/O waits and CPU bursts
- **Advantage:** Enables controlled experiments, reproducibility
- **Limitation:** May not capture complex real-world patterns
- **Best Practice:** Authors plan transition to real GWA-T-4 data for validation

### C. Differences in Approaches and Results

#### 1. Model Selection Philosophy

**Paper 1: Ensemble-Centric**
- **Philosophy:** Leverage ensemble learning for robustness
- **Primary Focus:** Random Forest, XGBoost (ensemble methods)
- **Result:** Ensemble (RF) wins

**Paper 2: Comprehensive Survey**
- **Philosophy:** Evaluate diverse algorithm families to find best
- **Coverage:** Distance-based (KNN), kernel (SVM), tree (DT, RF, XGB), neural (ANN)
- **Result:** Neural network (ANN) wins

**Paper 3: Pragmatic Selection**
- **Philosophy:** Choose established, interpretable methods
- **Coverage:** Classic ML (LR, KNN, DT) + one neural approach (MLP)
- **Result:** Decision Tree wins

**Interpretation:**
- No universal "best" algorithm—dataset characteristics matter
- Paper 2's synthetic data may favor flexible models (ANN)
- Paper 3's large real dataset enables complex tree structures
- Paper 1's moderate dataset size benefits from bagging (RF)

#### 2. Neural Network Performance Variance

**Stark Contrast:**
- **Paper 2:** ANN achieves best performance (MAE=4.13, CC=0.885)
- **Paper 3:** MLP achieves worst performance (26.01% accuracy)

**Potential Explanations:**
1. **Architecture Differences:**
   - Paper 2: Single hidden layer (100 nodes), tanh activation
   - Paper 3: 20 hidden layers—potentially overparameterized, vanishing gradients [39]

2. **Data Characteristics:**
   - Paper 2: Small synthetic dataset (1K samples) may favor simpler architecture
   - Paper 3: Large real dataset (347K samples) with complex tree structure better suited to Decision Trees

3. **Hyperparameter Tuning:**
   - Paper 2: GridSearchCV optimization
   - Paper 3: Fixed learning rate (0.001), no mention of optimization

4. **Regularization:**
   - Paper 2: L2 regularization (α=0.001)
   - Paper 3: Not mentioned

**Lesson:** Neural network success highly dependent on proper architecture selection and tuning.

#### 3. Feature Engineering Depth

**Minimal (Paper 2):**
- 7 original features → 3 selected + 1 engineered (io_to_burst_ratio)
- **Rationale:** Simplicity for real-time prediction
- **Trade-off:** May sacrifice some accuracy

**Moderate (Paper 1):**
- 29 original features → 11 selected
- **Rationale:** Balance between complexity and performance
- **Approach:** Statistical significance (Chi-Square)

**Extensive (Paper 3):**
- 29 original features → 9 selected through multi-step process
- **Rationale:** Maximize information retention from large dataset
- **Approach:** Feature importance scoring with threshold

**Insight:** Optimal feature count correlates with dataset size:
- Small dataset (Paper 2, 1K): 3-4 features
- Medium dataset (Paper 1, 30K): 11 features
- Large dataset (Paper 3, 347K): 8-9 features

Prevents overfitting on small datasets; captures complexity on large datasets.

#### 4. Evaluation Metric Focus

**Paper 1: Regression-Oriented**
- **Metrics:** MAE, RMSE, R²
- **Focus:** Prediction error magnitude and variance explained
- **Advantage:** Standard regression evaluation

**Paper 2: Comprehensive Regression + Correlation**
- **Metrics:** MAE, RMSE, RAE, CC
- **Focus:** Absolute error, relative improvement, linear relationship
- **Advantage:** RAE enables direct baseline comparison

**Paper 3: Classification-Style + Scheduling**
- **Metrics:** Accuracy percentage, avg TT, avg WT, CS
- **Focus:** End-to-end system performance
- **Advantage:** Directly measures scheduling efficiency

**Implication:** Different metrics serve different research goals:
- Papers 1 & 2: Prediction accuracy primary goal
- Paper 3: Scheduling algorithm performance primary goal

#### 5. Baseline Comparison Rigor

**Paper 1: Implicit**
- **Baseline:** Exponential Averaging mentioned but not experimentally evaluated
- **Limitation:** No quantitative improvement measurement

**Paper 2: Explicit Dual Baseline**
- **Baseline 1:** Exponential Averaging (MAE=20.67 ms)
- **Baseline 2:** Oracle (perfect predictions, MAE=0)
- **Advantage:** Bounds performance—shows ML achieves 80% of theoretical improvement

**Paper 3: Multi-Algorithm Comparison**
- **Baselines:** EA + 4 Round Robin variants (RR, SARR, MRRA, ORR)
- **Advantage:** Demonstrates superiority over multiple existing approaches
- **Strength:** 10-epoch consecutive test shows consistency

**Best Practice:** Paper 2's dual baseline (traditional method + theoretical optimal) provides most complete performance context.

#### 6. Scalability Demonstration

**Paper 1: Single-Scale**
- **Test:** 30,000 processes (fixed)
- **Limitation:** Scalability unknown

**Paper 2: Single-Scale**
- **Test:** 1,000 processes (fixed)
- **Limitation:** Performance on larger loads unclear

**Paper 3: Multi-Scale**
- **Tests:** 500, 1000, 1500, ..., 5000 processes (10 epochs)
- **Additional:** 10,000 process stress test
- **Advantage:** Demonstrates consistent performance advantage across scales
- **Insight:** ADRR's ~2x improvement maintained from 500 to 5,000 processes

**Conclusion:** Paper 3 provides strongest evidence of scalability.

#### 7. Scheduling Algorithm Integration

**Paper 1: Theoretical Only**
- **Discussion:** Mentions SJF/SRTF could use predictions
- **Implementation:** None
- **Gap:** Prediction accuracy not translated to scheduling performance

**Paper 2: Simulation Integration**
- **Implementation:** Python simulation engine
- **Algorithms:** SJF, SRTF with ML-predicted burst times
- **Metrics:** Waiting time, turnaround time
- **Limitation:** Simulation artifacts (unrealistic absolute values)

**Paper 3: Novel Algorithm with Integration**
- **Contribution:** New ADRR algorithm specifically designed for ML predictions
- **Implementation:** Full algorithm with dynamic time quantum
- **Comparison:** Head-to-head vs. 4 existing algorithms
- **Strength:** Demonstrates end-to-end system improvement

**Conclusion:** Paper 3 provides most complete scheduling integration; Paper 2 demonstrates SJF/SRTF feasibility; Paper 1 lacks integration.

---

## V. CONCLUSION

### A. Summary of Major Findings

This review analyzed three recent research contributions addressing CPU burst time prediction using Machine Learning to enhance process scheduling algorithms. The collective findings demonstrate:

#### 1. ML Superiority Over Traditional Methods

All three studies conclusively show that Machine Learning models significantly outperform traditional Exponential Averaging for CPU burst time prediction:

- **Paper 1 [1]:** Random Forest achieved R²=0.992 compared to EA's typical 20-30% accuracy
- **Paper 2 [2]:** ANN reduced prediction error by 80% (MAE: 4.13 ms vs. EA: 20.67 ms)
- **Paper 3 [3]:** Decision Tree achieved 98.64% accuracy vs. EA's 26.08%

**Consensus:** ML-based prediction is a viable alternative to heuristic methods, offering substantial accuracy improvements across diverse datasets and model architectures.

#### 2. Tree-Based Models Demonstrate Consistent Excellence

Ensemble tree-based algorithms emerged as top performers across multiple studies:

- **Paper 1:** Random Forest (best overall: MAE=14.150, R²=0.992)
- **Paper 3:** Decision Tree (best overall: 98.64% accuracy)
- **Paper 2:** Random Forest and XGBoost among top-5 models

**Insight:** Tree-based methods appear particularly well-suited to grid computing workloads, effectively capturing non-linear relationships between process attributes and burst times without requiring extensive feature engineering [26], [28].

#### 3. Feature Importance Convergence

Despite using different datasets and feature selection methods, all studies identified similar key predictors:

**Top Features Across Studies:**
1. **Previous/Average CPU Time Used:** Strongest predictor in all three papers
2. **Memory Usage:** Consistently among top-3 features
3. **I/O Wait Time:** Important in Papers 2 and others
4. **Requested Time:** Significant in Papers 1 and 3

**Implication:** Future implementations can focus on collecting and optimizing these critical features, potentially simplifying data collection infrastructure.

#### 4. Neural Networks: Architecture-Sensitive

Neural network performance varied dramatically:
- **Paper 2:** ANN achieved best results (MAE=4.13 ms, CC=0.885)
- **Paper 3:** MLP performed worst (26.01% accuracy)

**Lesson:** Success with neural networks requires careful architecture design, hyperparameter tuning, and regularization [34], [39]. Simple application of deep networks without optimization can underperform even traditional ML methods.

#### 5. Scheduling Integration Proves Feasibility

Papers 2 and 3 demonstrated that ML predictions translate to tangible scheduling improvements:

- **Paper 2:** 6.5% reduction in waiting/turnaround times vs. EA; within 5% of theoretical optimal
- **Paper 3:** ~54% reduction vs. traditional RR; ~2x performance vs. multiple algorithms

**Conclusion:** ML-based scheduling is not merely theoretically sound but practically implementable with measurable benefits.

#### 6. Scalability Maintained Across Workload Sizes

Paper 3's extensive testing (500-5,000 processes) showed consistent performance advantages, suggesting ML-based approaches scale effectively. However, real-time prediction latency in production environments remains unquantified across all studies.

#### 7. Implementation Gaps Persist

**Critical Finding:** All three papers rely on simulations rather than actual operating system implementations. Kernel-level integration, real-time performance, and production deployment challenges remain largely unexplored.

### B. How This Review Supports the Next Assignment (Implementation)

This comprehensive review provides essential foundation for subsequent implementation work in several critical ways:

#### 1. Algorithm Selection Guidance

**Recommended Primary Model:** **Random Forest or Decision Tree**

**Rationale:**
- Consistent top performance across all three studies
- High interpretability—can explain scheduling decisions [26], [28]
- Robust to overfitting with proper hyperparameter tuning
- Computationally efficient for real-time prediction
- Well-supported in standard ML libraries (Scikit-learn, XGBoost)

**Implementation Strategy:**
```python
# Recommended starting configuration based on review findings
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=55,      # Paper 1 optimal value
    max_depth=10,         # Paper 2 configuration
    criterion='squared_error',
    min_samples_split=2,
    min_samples_leaf=3,   # Paper 1 optimization
    random_state=42
)
```

**Alternative for Advanced Implementation:** Artificial Neural Networks if sufficient computational resources and tuning time available:
```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100,),  # Paper 2 configuration
    activation='tanh',
    alpha=0.001,                # L2