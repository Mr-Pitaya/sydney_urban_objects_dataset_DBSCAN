import tools
import pandas as pd

file_name = 'scan.23124.csv'
range_coordinate = (1,4)
raster_map_parameters = {'x_min': -15, 'x_max': 25,
                         'y_min': -15, 'y_max': 15,
                         'resolution': 1,}
dbscan_parameter = {'eps': 0.8, 'minpts': 20}
filter_parameter = {'delta_h': 0.3, 'ground_h': 0}


original_data = tools.coordinate_system(tools.data_processing(file_name, *range_coordinate))

pd.DataFrame(data = original_data).to_csv('original_data.csv', index= None, header= None, encoding= 'gbk')
print('已生成原始数据文件original_data.csv')

#raster_map = tools.raster_map(original_data, **raster_map_parameters)

raster_filter = tools.raster_map_filter(tools.raster_map(original_data, **raster_map_parameters), **filter_parameter)

filter_data = []
for i in raster_filter:
    filter_data.extend(i['data'])

pd.DataFrame(data = filter_data).to_csv('filter_data.csv', index= None, header= None, encoding= 'gbk')
print('已生成滤波后数据文件filter_data.csv')

raster_filter_result, label_max = tools.raster_clustering(raster_filter, raster_map_parameters)

dbscan_result = tools.local_dbscan(raster_filter, label_max, dbscan_parameter)
