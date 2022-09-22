import pyproj
#import psycopg2


def vn2k_to_wgs84(coordinate, crs):
    """
    Đây là hàm chuyển đổi cặp toạ độ x, y theo vn2k sang kinh độ , vĩ độ theo khung toạ độ của Google Map
    Công thức này được cung cấp bởi thư viện pyproj


    Input:

        - ( x, y ) : TUPLE chứa cặp toạ độ x và y theo đơn vị float
        - crs : INT - id (mã) vùng chứa cặp toạ độ x, y theo toạ độ Google

    Output:

        - (longitude, latitude): TUPLE chứa cặp kinh độ - vĩ độ theo toạ độ Google Map


    """
    new_coordinate = pyproj.Transformer.from_crs(
        crs_from=crs, crs_to=4326, always_xy=True).transform(coordinate[0], coordinate[1])

    return new_coordinate


# def vn2k_to_wgs84(coordinate, city_id):
#     """
#     Đây là hàm chuyển đổi cặp toạ độ x, y theo vn2k sang kinh độ , vĩ độ theo khung toạ độ của Google Map
#     Công thức này được cung cấp bởi thư viện pyproj


#     Input:

#         - Cặp toạ độ x, y : TUPLE chứa cặp toạ độ x và y theo đơn vị float
#         - city_id : INT - id (mã) vùng thành phố/ tỉnh mà chứa cặp toạ độ x, y theo toạ độ việt nam

#     Output:

#         - Cặp kinh độ - vĩ độ : TUPLE chứa cặp kinh độ - vĩ độ


#     """


#     crs = convert_to_crs('city_id')
#     if crs != -1:
#         new_coordinate = pyproj.Transformer.from_crs(
#             crs_from=crs, crs_to=4326, always_xy=True).transform(coordinate[0], coordinate[1])

#     else:
#        return (dict(value=-1, message='Cannot find any crs matching city_id'))
#     return new_coordinate


# def convert_to_crs(city_id):
#     try:
#         connection = psycopg2.connect(
#             user='ngocho',
#             password='123',
#             host='localhost',
#             port='5432',
#             dbname= 'sharklanddb'
#         )

#         cursor = connection.cursor()

#         #########################################
#         # Lấy crs từ city_id cung cấp
#         #########################################
#         cursor.execute(
#             "SELECT crs FROM public.city WHERE id = {}".format(city_id))
#         result = cursor.fetchall()
#         if len(result) == 1:
#             crs = result[0][0]
#         else:
#             crs = -1
#         return int(crs)

#     except psycopg2.Error as error:

#         return None
#     finally:
#         if connection:
#             cursor.close()
#             connection.close()
