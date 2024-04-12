## Project Diary - Work Distribution
**TO DO:** 
- Join the Competition and share your username with Wayne (he is the team leader and will add you to the group)
- Join our GitHub repository
- Clone the repository locally
- Download the data (parquet)
- Read the competition goals, metrics, and data provided...
- Extra credits: do some research on other similar projects and model stability - If you find a good article please add it to the Resources folder so everyone can learn from it ;)
- Decide your preferred development environment (E.j. Visual Studio Code)
- Load the data and get a general understanding of what is there

Next Catch-Up: Sunday 3rd March

**Catch-Up Summary 1:**
We decided to split the data among the team members, each of us will have a deeper understanding of the assigned files
- Credit bureau – Wayne
- Previous applications - Shradha
- Base - Shradha
- Debit card, deposit, other - Laura
- Person - Laura
- Static - Artjola
- Tax registry - Artjola

**TO DO:**

- EDA on the assigned train files
- Ideally, pull the data as in the template
- Compile a summary of insights on what you found or what you didn't find

Next Catch-Up Sunday 30th March

**Catch-Up Summary 2:**
There is a great number of missing data, documentation on features is not enough to understand the data, large number of columns per table. We conclude it is imposible to understand each feature to decide how to input 
the null values, if drop the null values and the best way of aggregating the information. 

Decision, use a LGBM model with the raw data plus various aggregation methods by features. Use the LGBM feature selection to define which features to focus on an enhance the pre-processing (cleaning and feature 
extraction) on those only. 

**TO DO:** 
Following the previously assigned files per team member: 
- Depth 0 tables join to the base table on the case_id field directly
- Depth 1 and depth 2 tables, aggregate by case_id using the following aggregation methods:
  - Numerical: min, max, mean, median, sum 
  - Categorical: mode, (columnname_category) one hot encoding with the sum of counts
  - Dates: min, max, distinct_count
- Output, a table per person where the PK is the case_id and we have all the static columns and the aggregated ones.
- Join the four tables and train a LGBM 
- Discuss in the next catch-up about the feature selection results and next steps.

Next Catch-Up Tuesday 2nd April


**Working Session:** 
April 6th. All team members meet for 2.5 hours.  We build a simple LGBM model and run the feature importance metric, excluding the categorical data. 

Findings: 
- there is a large number of features with >50 unique categories. Doing one hot encoding is not technically possible due to out-of-memory issues.
- There are duplicates in the features, for instance, the person's age.
- Feature importance elbow found in the feature 10.
- We could use sparse algorithms for the categorical.
- Probably because of the AUC metric, we do not need to focus that much on the unbalanced target, we will confirm this experimentally. 

**Next Steps session:** All members meeting Friday 12th May. 
It is decided to do a new aggregation of the data by splitting it by num_group1 = 0 and num_group1 != 0 (applicant vs non-applicant) and re-train the LGBM model to get the new feature importance. Wayne will be doing this part using the previous EDA files. 

Shradha, Artjola, and Laura will investigate how to deal with the categorical features and try to come up with some transformation we could use to keep the information but reduce sparsity and dimensionality. 

By Sunday 14th April, we will decide the next steps with the results of the new model. Do we want to build an applicant-only model? Keep all together? Multiple model different thresholds? 

