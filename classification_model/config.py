values_reward = [0, 2, 3, 5, 10]

portfolio_filepath = 'data/portfolio.json'
profile_filepath = 'data/profile.json'
transcript_filepath = 'data/transcript.json'

portfolio_outputpath = 'data/portfolio.csv'
profile_outputpath = 'data/profile.csv'
transcript_outputpath = 'data/transcript.csv'
final_filepath = 'data/starsbucks_data_eda.csv'
data_modelpath = 'data/starsbucks_data_model.csv'

label = 'event'

# Numerical and Categorical Features

numerical_features = ['time', 'amount', 'age', 'income', 'difficulty', 'duration']
categorical_features = ['gender', 'offer_type', 'web', 'email', 'social', 'mobile']
new_numerical_features = ['time', 'amount', 'age', 'income', 'difficulty',
                          'duration', 'number_channels', 'antiquity']
date_features = ['became_member_on']

features = ['time', 'amount', 'age', 'income', 'difficulty', 'duration',
            'number_channels', 'antiquity', 'gender_M', 'gender_F', 'gender_O',
            'offer_type_bogo', 'offer_type_discount', 'offer_type_informational',
            'web_1', 'web_0', 'email_1', 'email_0', 'social_1', 'social_0', 'mobile_1', 'mobile_0']

gender_categories = ['M', 'F', 'O']
offer_categories = ['bogo', 'discount', 'informational']
channel_categories = ['1', '0']
