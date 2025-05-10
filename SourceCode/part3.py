import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# Bài 3: Phân loại cầu thủ với KMeans + PCA (chọn k tự động qua Elbow detection)
# Thêm hiển thị centroids trên biểu đồ PCA
# -----------------------------

def detect_elbow(k_values, wcss):
    # Sử dụng phương pháp khoảng cách điểm tới đường thẳng nối đầu-cuối
    points = np.column_stack((k_values, wcss))
    first_point = points[0]
    last_point = points[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for p in points:
        vec = p - first_point
        proj = np.dot(vec, line_vec_norm) * line_vec_norm
        dist = np.linalg.norm(vec - proj)
        distances.append(dist)
    return k_values[int(np.argmax(distances))]


def cluster_players_part3(results_csv='SourceCode/results.csv', k_min=2, k_max=78):
    # 1. Đọc dữ liệu
    df = pd.read_csv(results_csv)

    # 2. Xử lý dữ liệu thiếu
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric.replace('N/a', np.nan, inplace=True)
    numeric.dropna(axis=1, how='all', inplace=True)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(numeric)

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 4. Tính WCSS và detect elbow
    ks = np.arange(k_min, k_max + 1)
    wcss = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    optimal_k = detect_elbow(ks, wcss)
    print(f"Tự động chọn k = {optimal_k}")

    # 5. Vẽ Elbow plot
    plt.figure(figsize=(8,4))
    plt.plot(ks, wcss, 'o-')
    plt.axvline(optimal_k, linestyle='--', label=f'Elbow tại k={optimal_k}')
    plt.xlabel('Số cụm k')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6. KMeans với k tối ưu
    kmeans_opt = KMeans(n_clusters=int(optimal_k), random_state=42)
    labels = kmeans_opt.fit_predict(X_scaled)
    centers_scaled = kmeans_opt.cluster_centers_

    # 7. PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(centers_scaled)

    # 8. Vẽ scatter phân cụm và centroids
    plt.figure(figsize=(10,6))
    for cid in range(int(optimal_k)):
        plt.scatter(
            X_pca[labels == cid, 0], X_pca[labels == cid, 1],
            label=f'Cụm {cid+1}', alpha=0.6
        )
    # Vẽ centroids
    plt.scatter(
        centers_pca[:,0], centers_pca[:,1],
        s=200, c='black', marker='X', label='Centroids'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'KMeans Clustering (k={int(optimal_k)}) với Centroids')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return labels, int(optimal_k)

if __name__ == '__main__':
    labels, k = cluster_players_part3()
    print(f"Hoàn thành phân cụm với k = {k}")
