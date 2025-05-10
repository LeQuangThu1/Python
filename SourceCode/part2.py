import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Đọc dữ liệu từ file CSV
df = pd.read_csv('SourceCode/results.csv')

# Đảm bảo cột 'Minutes' là kiểu chuỗi rồi thay thế dấu phẩy, sau đó chuyển thành kiểu float
df['Minutes'] = df['Minutes'].astype(str).str.replace(',', '').astype(float)

# Lọc dữ liệu chỉ bao gồm cầu thủ chơi trên 90 phút
df_filtered = df[df['Minutes'] > 90]

# 1. **Xác định top 3 cầu thủ cao nhất và thấp nhất cho mỗi thống kê**

# Đảm bảo các cột thống kê có kiểu số (float)
for column in df_filtered.columns:
    if column not in ['Name', 'Team', 'Position', 'Nation', 'Age']:
        # Chuyển các cột không phải tên, đội, vị trí, quốc gia, tuổi thành kiểu số
        df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')

# 1. **Xác định top 3 cầu thủ cao nhất và thấp nhất cho mỗi thống kê**

with open('SourceCode/top_3.txt', 'w', encoding = 'utf-8') as file:
    # Lặp qua tất cả các cột thống kê trong df_filtered
    for column in df_filtered.columns:
        # Kiểm tra nếu cột là một cột thống kê, không phải cột như "Name", "Team", "Position"...
        if column not in ['Name', 'Team', 'Position', 'Nation', 'Age']:
            # Lấy top 3 cầu thủ có điểm cao nhất và thấp nhất
            top_3_highest = df_filtered.nlargest(3, column)[['Name', column]]
            top_3_lowest = df_filtered.nsmallest(3, column)[['Name', column]]
            
            # Ghi vào file top_3.txt
            file.write(f"Top 3 cầu thủ có {column} cao nhất:\n")
            file.write(top_3_highest.to_string(index=False))
            file.write("\n\n")
            file.write(f"Top 3 cầu thủ có {column} thấp nhất:\n")
            file.write(top_3_lowest.to_string(index=False))
            file.write("\n\n")


# 2. **Tính toán Trung vị (Median), Trung bình (Mean), và Độ lệch chuẩn (Std)**

results = []
results.append(["Team", "Statistic", "Median of Statistic", "Mean of Statistic", "Std of Statistic"])

# Lặp qua tất cả các cột thống kê và nhóm theo "Team" để tính giá trị thống kê cho từng đội
for column in df_filtered.columns:
    if column not in ['Name', 'Team', 'Position', 'Nation', 'Age']:
        # Tính toán thống kê cho từng đội
        grouped = df_filtered.groupby('Team')[column]
        median_val = grouped.median()
        mean_val = grouped.mean()

        # Kiểm tra độ lệch chuẩn (std) với ít nhất 2 giá trị hợp lệ
        std_val = grouped.std()
        
        # Nếu có ít hơn 2 giá trị hợp lệ, thay thế std bằng NaN hoặc 0
        std_val = std_val.fillna(0)  # Hoặc std_val = std_val.fillna(np.nan) nếu bạn muốn giữ NaN

        # Lưu các kết quả vào danh sách
        for team in median_val.index:
            results.append([team, column, median_val[team], mean_val[team], std_val[team]])

# Lưu kết quả vào file results2.csv
df_results = pd.DataFrame(results[1:], columns=results[0])
df_results.to_csv('SourceCode/results2.csv', index=False)

# 3. **Vẽ biểu đồ Histogram cho từng thống kê**
# Đọc dữ liệu từ file CSV
df = pd.read_csv('SourceCode/results.csv')

# Đảm bảo cột 'Minutes' là kiểu chuỗi rồi thay thế dấu phẩy, sau đó chuyển thành kiểu float
df['Minutes'] = df['Minutes'].astype(str).str.replace(',', '').astype(float)

# Lọc dữ liệu chỉ bao gồm cầu thủ chơi trên 90 phút
df_filtered = df[df['Minutes'] > 90]

# Các chỉ số thống kê bạn muốn vẽ biểu đồ cho
columns_to_plot = ['Goals', 'Assists', 'xG', 'Goals against per 90', 'Tkl', 'Int']

# Kiểm tra nếu thư mục để lưu hình ảnh và PDF đã tồn tại, nếu không tạo mới
output_dir = 'SourceCode/image'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tạo đối tượng PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Thêm tiêu đề
pdf.set_font("Arial", size=12, style='B')
pdf.cell(200, 10, txt="Histograms for Player Statistics", ln=True, align='C')
pdf.ln(10)  # Dấu xuống dòng

# 1. Vẽ histogram cho tất cả cầu thủ trong giải đấu
for column in columns_to_plot:
    # Vẽ biểu đồ histogram cho tất cả cầu thủ
    plt.figure(figsize=(8, 6))
    plt.hist(df_filtered[column].dropna(), bins=20, edgecolor='black')
    plt.title(f"Distribution of {column} for All Players")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Lưu biểu đồ vào tệp PNG trong thư mục đã tạo
    image_filename = os.path.join(output_dir, f'histogram_all_players_{column}.png')
    plt.savefig(image_filename)
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
    
    # Thêm hình ảnh vào PDF
    pdf.add_page()  # Tạo một trang mới
    pdf.image(image_filename, x=10, y=30, w=180)  # Thêm hình ảnh vào PDF
    pdf.ln(10)  # Dấu xuống dòng

# 2. Vẽ histogram cho từng đội
for column in columns_to_plot:
    for team in df_filtered['Team'].unique():
        team_data = df_filtered[df_filtered['Team'] == team][column]
        
        # Vẽ biểu đồ histogram cho từng đội
        plt.figure(figsize=(8, 6))
        plt.hist(team_data.dropna(), bins=20, edgecolor='black')
        plt.title(f"Distribution of {column} for {team}")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Lưu biểu đồ vào tệp PNG trong thư mục đã tạo
        image_filename = os.path.join(output_dir, f'histogram_{team}_{column}.png')
        plt.savefig(image_filename)
        plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
        
        # Thêm hình ảnh vào PDF
        pdf.add_page()  # Tạo một trang mới
        pdf.image(image_filename, x=10, y=30, w=180)  # Thêm hình ảnh vào PDF
        pdf.ln(10)  # Dấu xuống dòng

# Lưu PDF
pdf_output_filename = os.path.join(output_dir, "player_statistics_histograms.pdf")
pdf.output(pdf_output_filename)

print(f"PDF saved as {pdf_output_filename}")


# 4. **Xác định đội bóng có số điểm cao nhất cho mỗi thống kê**
# Đọc lại dữ liệu từ results2.csv để lấy thông tin đội bóng
df = pd.read_csv('SourceCode/results2.csv')

# Kiểm tra tên các cột trong DataFrame
# print(df.columns)  # In ra tên các cột để kiểm tra

# Loại bỏ khoảng trắng thừa trong tên cột
df.columns = df.columns.str.strip()

# Kiểm tra xem có cột "Team" không
if 'Team' not in df.columns:
    print("Cột 'Team' không tồn tại trong dữ liệu!")
else:
    # Lọc dữ liệu bỏ đi đội có tên 'all'
    df_teams = df[df['Team'] != 'all'].copy()

    # Lọc ra các cột có tên bắt đầu bằng "Mean of" (tính trung bình của các thống kê)
    mean_columns = [col for col in df_teams.columns if col.startswith("Mean of")]

    # Danh sách lưu thông tin về các đội có số điểm cao nhất cho mỗi thống kê
    top_teams_info = []

    # Duyệt qua các cột thống kê trung bình để tìm đội có giá trị cao nhất
    for column in mean_columns:
        # Lấy dòng có giá trị cao nhất trong cột thống kê
        best_team_row = df_teams.loc[df_teams[column].idxmax()]
        
        # Lưu thông tin đội có số điểm cao nhất cho thống kê này
        top_teams_info.append({
            'Statistic': column.replace("Mean of ", ""),  # Loại bỏ "Mean of" khỏi tên cột
            'Top Team': best_team_row['Team'],
            'Value': best_team_row[column]
        })

    # Tạo DataFrame từ danh sách top_teams_info
    df_top_teams = pd.DataFrame(top_teams_info)

    # In ra bảng kết quả các đội có số điểm cao nhất cho mỗi thống kê
    # print(df_top_teams)

    # Tính số lần mỗi đội đứng đầu trong danh sách "Top Team" (tức là đội chiến thắng nhiều thống kê)
    team_performance_counts = df_top_teams['Top Team'].value_counts()

  
    best_team = team_performance_counts.idxmax()

    # In ra đội bóng tốt nhất dựa trên số lần đứng đầu các thống kê
    print(f"\nBest performing team based on overall means: {best_team}")