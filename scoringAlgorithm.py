import numpy as np
import math
import pandas as pd
from collections import OrderedDict

class ScoringAlgorithm:

    def __init__(self, path_target, path_input, window_size=100):
        self.df_target = self.read_data(path_target)
        self.df_input  = self.read_data(path_input)

        self.window_size_scoring_ = window_size
        self.padding_ = self.window_size_scoring_ // 4
        data_size_target = len(self.df_target.index)
        data_size_input  = len(self.df_input.index)
        self.data_size_scoring_ = (data_size_target if data_size_target < data_size_input else data_size_input) - self.padding_ - 10


    ''' テスト用データ作成 '''
    def read_data(self, path):
        import ast
        with open(path) as f:
            list_d = [ast.literal_eval(s.split(':', 1)[1]) for s in f.readlines()]

        df_data = pd.DataFrame(index=range(len(list_d)), columns=[])
        df_data["RShoulder_x"] = [dic_tmp[2][0] if 2 in dic_tmp else np.nan for dic_tmp in list_d]
        df_data["RShoulder_y"] = [dic_tmp[2][1] if 2 in dic_tmp else np.nan for dic_tmp in list_d]
        df_data["RElbow_x"]    = [dic_tmp[3][0] if 3 in dic_tmp else np.nan for dic_tmp in list_d]
        df_data["RElbow_y"]    = [dic_tmp[3][1] if 3 in dic_tmp else np.nan for dic_tmp in list_d]


        # # txtファイルを文字列として読み込み、辞書型として読み込めるように加工
        # with open(path) as f:
        #     str_data = ''.join([ '\"' + s.split(':', 1)[0] + '\"' + ':' + s.split(":", 1)[1] + ','  for s in f.readlines()])
        #     str_data_json = '{' + str_data.replace(" ", "") + '}'
        #
        # import ast
        # dic_data = ast.literal_eval(str_data_json) # 辞書型への変換
        #
        # df_data = pd.DataFrame(index=range(len(dic_data.keys())), columns=[])
        # # print(type(dic_data))
        # # print(dic_data)
        # df_data["RShoulder_x"] = [dic_tmp[2][0] if 2 in dic_tmp else np.nan for _, dic_tmp in dic_data.items()]
        # df_data["RShoulder_y"] = [dic_tmp[2][1] if 2 in dic_tmp else np.nan for _, dic_tmp in dic_data.items()]
        # df_data["RElbow_x"]    = [dic_tmp[3][0] if 3 in dic_tmp else np.nan for _, dic_tmp in dic_data.items()]
        # df_data["RElbow_y"]    = [dic_tmp[3][1] if 3 in dic_tmp else np.nan for _, dic_tmp in dic_data.items()]

        return df_data


    def score(self):
        df_preprocessed_target = self.preprocess(self.df_target)
        df_preprocessed_input  = self.preprocess(self.df_input)

        arr_rotation_target = self.calc_rotations(df_preprocessed_target["RShoulder_x"].values, df_preprocessed_target["RShoulder_y"].values, df_preprocessed_target["RElbow_x"].values, df_preprocessed_target["RElbow_y"].values)
        arr_rotation_input  = self.calc_rotations(df_preprocessed_input["RShoulder_x"].values,  df_preprocessed_input["RShoulder_y"].values,  df_preprocessed_input["RElbow_x"].values,  df_preprocessed_input["RElbow_y"].values)

        dic_rotation_target_windowed = self.window_target_rotation(arr_rotation_target)
        print('size_target_windowed = ' + str(len(dic_rotation_target_windowed)))
        if(len(dic_rotation_target_windowed) < 1):
            return 0
        dic_rotation_input_windowed = self.window_input_rotation(arr_rotation_input, dic_rotation_target_windowed)
        print('size_input_windowed = ' + str(len(dic_rotation_input_windowed)))

        list_score_physical_strength = []
        for key in dic_rotation_input_windowed.keys():
            print('key score = ' + str(key))
            score_physical_strength, _ = self.calc_score(dic_rotation_target_windowed[key], dic_rotation_input_windowed[key])
            list_score_physical_strength.append(score_physical_strength)

        if(len(list_score_physical_strength)) < 1:
            print("Error : zero dicision")
            return 0
        score_result = math.floor(sum(list_score_physical_strength) / len(list_score_physical_strength))

        return score_result

    def filter_median(self, arr_data):
        window_size_filter_ = 3
        median_offset = window_size_filter_ // 2
        arr_data = np.array(arr_data)
        return  [ np.median(arr_data[i-median_offset : i+median_offset+1]) for i in range(median_offset, len(arr_data)-median_offset) ]


    def preprocess(self, df_src):
        df_tmp = df_src.interpolate(limit_direction='both')
        df_dst = pd.DataFrame(index=range(len( self.filter_median(df_tmp.index) )), columns=df_tmp.columns)
        for seg_name in df_tmp.columns:
            df_dst[seg_name] = self.filter_median(df_tmp[seg_name].values)
        return df_dst

    def calc_rotations(self, arr_pos_x_target, arr_pos_y_target, arr_pos_x_input, arr_pos_y_input):
        pos_diff_x = arr_pos_x_input - arr_pos_x_target
        pos_diff_y = arr_pos_y_input - arr_pos_y_target
        arr_rotation = np.array([math.atan2(pos_diff_x[i], pos_diff_y[i]) * 180 / math.pi for i in range(pos_diff_x.shape[0])])
        return arr_rotation


    def window_target_rotation(self, arr_rotation):
        thresh_is_motion_ = 20

        list_rotation_extracted = []
        dic_rotation_extracted = {}
        for i_window in range( self.data_size_scoring_ //  self.window_size_scoring_):
            first_id = i_window * self.window_size_scoring_ + self.padding_
            list_slice = range(first_id, first_id + self.window_size_scoring_)
            arr_rotation_windowed = arr_rotation[list_slice]

            # if abs(np.median(arr_rotation_windowed)) > thresh_is_motion_:
                # list_rotation_extracted.append(arr_rotation_windowed)
                # dic_rotation_extracted[first_id] = arr_rotation_windowed
            list_rotation_extracted.append(arr_rotation_windowed)
            dic_rotation_extracted[first_id] = arr_rotation_windowed

        return dic_rotation_extracted


    def window_input_rotation(self, arr_rotation_input, dic_target):
        dic_rotation_extracted = {}
        for i_first, arr_rotation_target_windowed in dic_target.items():
            # self.padding_ = self.window_size_scoring_ // 4
            list_slice = range(i_first - self.padding_, i_first + self.window_size_scoring_ + self.padding_)
            tmp_arr_rotation_input   = arr_rotation_input[list_slice]
            tmp_arr_rotation_target = dic_target[i_first]

            # 相互相関係数が最も高いdelayを算出
            zero_rotation_target = tmp_arr_rotation_target - tmp_arr_rotation_target.mean()
            zero_rotation_input   = tmp_arr_rotation_input   - tmp_arr_rotation_input.mean()
            corr = np.correlate(zero_rotation_target, tmp_arr_rotation_input, "full")
            delay = -(corr.argmax() - zero_rotation_input.shape[0] + 1)
            if abs(delay) > self.padding_:
                delay = 0
            print('delay = ' + str(delay))

            dic_rotation_extracted[i_first] = tmp_arr_rotation_input[delay : delay + self.window_size_scoring_]

        return dic_rotation_extracted


    def calc_score(self, arr_rotaion_target, arr_rotaion_input):
        # 筋力
        peak_value_target = abs(arr_rotaion_target).max()
        peak_value_input = abs(arr_rotaion_input).max()
        peak_value_max = peak_value_target if peak_value_target > peak_value_input else peak_value_input
        score_physical_strength = 100 * (1 - (abs(peak_value_input - peak_value_target) / peak_value_max))

        # 類似度
        arr_rotaion_target_zero = arr_rotaion_target - arr_rotaion_target.mean()
        arr_rotaion_input_zero = arr_rotaion_input - arr_rotaion_input.mean()
        score_corr = np.correlate(arr_rotaion_target_zero, arr_rotaion_input_zero)

        return score_physical_strength, score_corr



# print(ScoringAlgorithm('../resource/target_long.txt', '../resource/input.txt', 60).score())