import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

# Set up Selenium with Microsoft Edge WebDriver
edge_options = Options()
edge_options.add_argument("--headless")  # Run headless
edge_options.add_argument("--disable-gpu")  # Disable GPU

# Specify the path to msedgedriver
edge_driver_path = r'I:/edgedriver_win64/msedgedriver.exe'

# Initialize the driver
try:
    service = Service(edge_driver_path)
    driver = webdriver.Edge(options=edge_options)
except Exception as e:
    print(f"Error: {e}")
    print("Please check the msedgedriver installation and path.")
    exit()

# List of URLs to scrape
urls = [
    'https://fbref.com/en/comps/9/stats/Premier-League-Stats',
    'https://fbref.com/en/comps/9/keepers/Premier-League-Stats',
    'https://fbref.com/en/comps/9/shooting/Premier-League-Stats',
    'https://fbref.com/en/comps/9/passing/Premier-League-Stats',
    'https://fbref.com/en/comps/9/gca/Premier-League-Stats',
    'https://fbref.com/en/comps/9/defense/Premier-League-Stats',
    'https://fbref.com/en/comps/9/possession/Premier-League-Stats',
    'https://fbref.com/en/comps/9/misc/Premier-League-Stats'
]

table = [
    'table#stats_standard',
    'table#stats_keeper',
    'table#stats_shooting',
    'table#stats_passing',
    'table#stats_gca',
    'table#stats_defense',
    'table#stats_possession',
    'table#stats_misc',
]

# Indices for each table
std_ls = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 16, 17, 18, 20, 22, 23, 24, 25, 26, 30, 31]
keepers_ls = [0, 11, 14, 19, 24]
shooting_ls = [0, 10, 12, 13, 15]
passing_ls = [0, 7, 9, 10, 14, 17, 20, 25, 26, 27, 28, 29]
gca_ls = [0, 8, 9, 16, 17]
defense_ls = [0, 7, 8, 13, 15, 16, 17, 18, 19]
possession_ls = [0, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28]
miscellanous_ls = [0, 10, 11, 12, 13, 19, 20, 21, 22]

ls = [
    std_ls,
    keepers_ls,
    shooting_ls,
    passing_ls,
    gca_ls,
    defense_ls,
    possession_ls,
    miscellanous_ls
]

# Function to scrape and process data from each URL
def scrape_stats(url, table_name, ls):
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    try:
        table = driver.find_element(By.CSS_SELECTOR, table_name)
        rows = table.find_elements(By.CSS_SELECTOR, 'tr:not(.thead)')
        player_data = []

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, 'td')
            if len(cells) > 1:
                row_data = [cells[index].text if index < len(cells) else None for index in ls]
                player_data.append(row_data)

        return player_data
    except Exception as e:
        print(f"Error scraping data from {url}: {e}")
        return []

# Collect data from all URLs
all_player_data = []
for i, url in enumerate(urls):
    data = scrape_stats(url, table[i], ls[i])
    all_player_data.append(data)

# Create DataFrames for each URL's data
try:
    df = pd.DataFrame(all_player_data[0], columns=["Name", "Nation", "Position", "Team", "Age", "Matches Played", "Starts", "Minutes", "Goals", "Assists", "Yellow Cards", "Red Cards", "xG", "xAG", "PrgC", "PrgP", "PrgR", "Goals per 90", "Assists per 90", "xG per 90", "xAG per 90"])
    keepers_df = pd.DataFrame(all_player_data[1], columns=["Name", "Goals against per 90", "Save%", "Clean Sheets %", "Penalty Save %"])
    shooting_df = pd.DataFrame(all_player_data[2], columns=["Name", "Shoots on target percentage", "Shoot on target per 90", "Goals per shot", "Average shoot distance"])
    passing_df = pd.DataFrame(all_player_data[3], columns=["Name", "Pass completed", "Pass completion", "Progressive passing distance", "Short pass completion", "Medium pass completion", "Long pass completion", "Key passes", "Pass in to final third", "Pass in to penalty are", "CrsPA", "PrgP"])
    gca_df = pd.DataFrame(all_player_data[4], columns=["Name", "SCA", "SCA90", "GCA", "GCA90"])
    defense_df = pd.DataFrame(all_player_data[5], columns=["Name", "Tkl", "TklW", "Att", "Lost", "Blocks", "Sh", "Pass", "Int"])
    possession_df = pd.DataFrame(all_player_data[6], columns=["Name", "Touches", "Def pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att pen", "att", "Succ%", "Tkld%", "Carries", "ProDist", "ProgC", "1/3", "CPA", "Mis", "Dis", "Rec", "PrgR"])
    miscellanous_df = pd.DataFrame(all_player_data[7], columns=["Name", "Fls", "Fld", "Off", "Crs", "Recov", "Won", "Lost", "Won%"])
except Exception as e:
    print(f"Error creating DataFrames: {e}")
    driver.quit()
    exit()

# Filter players with more than 90 minutes
df['Minutes'] = df['Minutes'].str.replace(',', '').astype(float)
df_filtered = df[df['Minutes'] > 90]

# Merge with other DataFrames
dfs = [keepers_df, shooting_df, passing_df, gca_df, defense_df, possession_df, miscellanous_df]
for dfi in dfs:
    dfi = dfi.drop_duplicates(subset='Name')
    df_filtered = df_filtered.merge(dfi, on='Name', how='left')

# Giả sử bạn đã có DataFrame df_filtered chứa các cầu thủ
# Tách tên đầu và tên họ (First Name và Last Name) từ cột "Name"
df_filtered['First Name'] = df_filtered['Name'].apply(lambda x: x.split()[0])
df_filtered['Last Name'] = df_filtered['Name'].apply(lambda x: x.split()[-1])

# Sắp xếp theo First Name, rồi đến Last Name
df_filtered = df_filtered.sort_values(by=['First Name', 'Last Name'])

# Xóa các cột "First Name" và "Last Name" tạm thời nếu không cần thiết trong output
df_filtered.drop(columns=['First Name', 'Last Name'], inplace=True)



# Handle missing values and save to CSV
df_filtered.fillna('N/a', inplace=True)
df_filtered.to_csv('SourceCode/results.csv', index=False)

# Close the browser
driver.quit()