README
======

    GOAL 
    ----
        Strengthen the capacity of all countries, in particular developing countries, for early warning, risk reduction and management of national and global health risks (Section 3D of sustainable goal development)

    HOW 
    ---
        Analyse location filtered health tweets, check for symptoms and verify predictive powers of social media(twitter) by using symptom mention in tweets to predict disease outbreak.


    DATASET
    -------
        1. 'tweet_all.txt' : Twitter data for the year 2017 for USA. Obtained by tweepy.
        2. 'hospital_data.tsv' : Patient intake data for hospitals across Texas. Includes columns for probable diagnosis.
        3. 'health_train_data.csv' : Health data corpus. It consists of text from medical articles. We use this to perform 'Topic modeling' to filter out health        related tweets.

    STEPS
    -----
        1. train.py : Uses health corpus to output health tweets into 'filteredtweet.txt'
        2. Run 'spark_symptom_countmatrix.py' file on filtered tweets to generate 'hepatitisA_count_matrix.csv' and 'influenza_count_matrix.csv'
        3. Run the 'spark_hospitaldata_count.py' file to generate 'influenza_hospital.csv' and 'hepatitisA_hospital.csv'
        4. Now run 'influenza_inference.py' file to find the correlation(linear regression model)
        5. Now run 'hepatitis_inference.py' file to find the correlation(linear regression model)

    CONTRIBUTORS
    ------------
        - Debesh Mohanty (111973815)
        - Rohit Bassi (112505198)
        - Shoaib Sheriff (111986606)