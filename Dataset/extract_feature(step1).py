import pandas as pd
import ast
import os

BASE_DIR = os.getcwd()

geo = pd.read_csv(os.path.join(BASE_DIR, 'MyModel/raw-data/foursquare_tky/foursquare_tky.geo'))
dyna = pd.read_csv(os.path.join(BASE_DIR, 'MyModel/raw-data/foursquare_tky/foursquare_tky.dyna'))

def get_geo_processed():
    # 解析 `coordinates` 列，转换为经度和纬度
    geo['coordinates'] = geo['coordinates'].apply(ast.literal_eval)  # 将字符串转换为列表
    geo['longitude'] = geo['coordinates'].apply(lambda x: x[0])  # 提取经度
    geo['latitude'] = geo['coordinates'].apply(lambda x: x[1])  # 提取纬度

    # 删除原始的 `coordinates` 列
    geo_data = geo.drop('coordinates', axis=1)

    # 地点去重
    geo_unique = geo_data.loc[
        geo_data.groupby(['type', 'venue_category_id', 'venue_category_name', 'longitude', 'latitude'])[
            'geo_id'].idxmin()]
    geo_data = geo_unique.sort_values(by='geo_id', ascending=True)

    # # 如果需要对 `venue_category_name` 进行编码，可以使用 pd.get_dummies 或 LabelEncoder
    # geo_data = pd.get_dummies(geo_data, columns=['venue_category_name'], prefix='category')
    return geo_data

def get_dyna_processed():
    # 将 time 列转换为 datetime 类型
    dyna['time'] = pd.to_datetime(dyna['time'], format='%Y-%m-%dT%H:%M:%SZ')

    # 提取时间特征
    dyna['hour'] = dyna['time'].dt.hour  # 小时
    dyna['weekday'] = dyna['time'].dt.weekday  # 星期几，0=周一，1=周二，...，6=周日
    dyna['month'] = dyna['time'].dt.month  # 月份
    dyna['day'] = dyna['time'].dt.day  # 日期
    dyna['is_weekend'] = dyna['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 周末标记
    return dyna

def get_merged_data():
    dyna_processed = get_dyna_processed()
    geo_processed = get_geo_processed()
    # 通过 `location` 将 `dyna` 与 `geo` 数据合并，得到地理信息
    dyna_data = dyna_processed.merge(geo_processed, how='left', left_on='location', right_on='geo_id')

    # 找出在 geo_processed 中未找到对应数据的 location
    missing_locations = dyna_data[dyna_data['geo_id'].isna()]['location']
    # 根据 location 去原始 geo 中找对应的数据
    for loc in missing_locations:
        matching_geo = geo[geo['geo_id'] == loc]
        match_geo = geo_processed[
            (geo_processed['venue_category_id'] == matching_geo['venue_category_id'].iloc[0]) &
            (geo_processed['venue_category_name'] == matching_geo['venue_category_name'].iloc[0]) &
            (geo_processed['longitude'] == matching_geo['longitude'].iloc[0]) &
            (geo_processed['latitude'] == matching_geo['latitude'].iloc[0])].iloc[0]
        dyna_data.loc[dyna_data['location'] == loc, 'geo_id'] = matching_geo['geo_id'].iloc[0]
        dyna_data.loc[dyna_data['location'] == loc, 'venue_category_id'] = match_geo['venue_category_id']
        dyna_data.loc[dyna_data['location'] == loc, 'venue_category_name'] = match_geo['venue_category_name']
        dyna_data.loc[dyna_data['location'] == loc, 'longitude'] = match_geo['longitude']
        dyna_data.loc[dyna_data['location'] == loc, 'latitude'] = match_geo['latitude']

    dyna_data = dyna_data.drop(['type_x', 'type_y', 'location'], axis=1)
    dyna_data['geo_id'] = dyna_data['geo_id'].astype(int)
    return dyna_data

merged_data = get_merged_data()
merged_data.to_csv(os.path.join(BASE_DIR, 'MyModel/data-process/foursquare_tky_processed.csv'), index=False)