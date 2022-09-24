
## Aicadium ML interview project 

**Data:**

Contains records for E-commerce transactions for online store
- Total of 12330 user transaction out of which 10422 are negative class (did not buy) and 1908 are positive class (bought product). "Revenue" attribute is the target feature.

- The dataset consists of 10 numerical and 8 categorical attributes.

- "Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration": These represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories. The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another.

- The "Bounce Rate", "Exit Rate" and "Page Value" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site.
    - Bounce Rate: The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session.
    - Exit Rate: The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session.
    - Page Value: The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction.

- Special Day: The "Special Day" feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction.
The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date.
For example, for Valentina’s day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8.

- The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.

The goal of this project is to build a ML model to predict whether or not a customer will buy the product.

**The pipeline includes:**
- Data exploration and statistics to understand the data, attributes and how they interact with each other.
- Feature scaling/normalization
- Feature importance and selection (also dealing with imbalanced dataset)
- model training and hyperparameter tuning
- model evaluation and metrics

