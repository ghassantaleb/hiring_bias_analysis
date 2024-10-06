{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "32d98b2c-6d8c-467a-afb0-b0bfd75e5d9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2b00074a-ac11-467b-b323-91764643c299",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('C:/Users/ghass/hiring_bias_analysis/Datasets/data job posts.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f3b7607b-7ce5-4cef-89f7-7795b89be86b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                             jobpost          date  \\\n",
      "0  AMERIA Investment Consulting Company\\r\\nJOB TI...   Jan 5, 2004   \n",
      "1  International Research & Exchanges Board (IREX...   Jan 7, 2004   \n",
      "2  Caucasus Environmental NGO Network (CENN)\\r\\nJ...   Jan 7, 2004   \n",
      "3  Manoff Group\\r\\nJOB TITLE:  BCC Specialist\\r\\n...   Jan 7, 2004   \n",
      "4  Yerevan Brandy Company\\r\\nJOB TITLE:  Software...  Jan 10, 2004   \n",
      "\n",
      "                                               Title  \\\n",
      "0                            Chief Financial Officer   \n",
      "1  Full-time Community Connections Intern (paid i...   \n",
      "2                                Country Coordinator   \n",
      "3                                     BCC Specialist   \n",
      "4                                 Software Developer   \n",
      "\n",
      "                                           Company AnnouncementCode Term  \\\n",
      "0             AMERIA Investment Consulting Company              NaN  NaN   \n",
      "1  International Research & Exchanges Board (IREX)              NaN  NaN   \n",
      "2        Caucasus Environmental NGO Network (CENN)              NaN  NaN   \n",
      "3                                     Manoff Group              NaN  NaN   \n",
      "4                           Yerevan Brandy Company              NaN  NaN   \n",
      "\n",
      "  Eligibility Audience StartDate                               Duration  ...  \\\n",
      "0         NaN      NaN       NaN                                    NaN  ...   \n",
      "1         NaN      NaN       NaN                               3 months  ...   \n",
      "2         NaN      NaN       NaN  Renewable annual contract\\r\\nPOSITION  ...   \n",
      "3         NaN      NaN       NaN                                    NaN  ...   \n",
      "4         NaN      NaN       NaN                                    NaN  ...   \n",
      "\n",
      "  Salary                                       ApplicationP OpeningDate  \\\n",
      "0    NaN  To apply for this position, please submit a\\r\\...         NaN   \n",
      "1    NaN  Please submit a cover letter and resume to:\\r\\...         NaN   \n",
      "2    NaN  Please send resume or CV toursula.kazarian@......         NaN   \n",
      "3    NaN  Please send cover letter and resume to Amy\\r\\n...         NaN   \n",
      "4    NaN  Successful candidates should submit\\r\\n- CV; \\...         NaN   \n",
      "\n",
      "                                        Deadline Notes  \\\n",
      "0                                26 January 2004   NaN   \n",
      "1                                12 January 2004   NaN   \n",
      "2  20 January 2004\\r\\nSTART DATE:  February 2004   NaN   \n",
      "3      23 January 2004\\r\\nSTART DATE:  Immediate   NaN   \n",
      "4                         20 January 2004, 18:00   NaN   \n",
      "\n",
      "                                              AboutC Attach  Year Month     IT  \n",
      "0                                                NaN    NaN  2004     1  False  \n",
      "1  The International Research & Exchanges Board (...    NaN  2004     1  False  \n",
      "2  The Caucasus Environmental NGO Network is a\\r\\...    NaN  2004     1  False  \n",
      "3                                                NaN    NaN  2004     1  False  \n",
      "4                                                NaN    NaN  2004     1   True  \n",
      "\n",
      "[5 rows x 24 columns]\n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e1ca77f6-8457-4ecd-8b2b-977c29bafbd0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 19001 entries, 0 to 19000\n",
      "Data columns (total 24 columns):\n",
      " #   Column            Non-Null Count  Dtype \n",
      "---  ------            --------------  ----- \n",
      " 0   jobpost           19001 non-null  object\n",
      " 1   date              19001 non-null  object\n",
      " 2   Title             18973 non-null  object\n",
      " 3   Company           18994 non-null  object\n",
      " 4   AnnouncementCode  1208 non-null   object\n",
      " 5   Term              7676 non-null   object\n",
      " 6   Eligibility       4930 non-null   object\n",
      " 7   Audience          640 non-null    object\n",
      " 8   StartDate         9675 non-null   object\n",
      " 9   Duration          10798 non-null  object\n",
      " 10  Location          18969 non-null  object\n",
      " 11  JobDescription    15109 non-null  object\n",
      " 12  JobRequirment     16479 non-null  object\n",
      " 13  RequiredQual      18517 non-null  object\n",
      " 14  Salary            9622 non-null   object\n",
      " 15  ApplicationP      18941 non-null  object\n",
      " 16  OpeningDate       18295 non-null  object\n",
      " 17  Deadline          18936 non-null  object\n",
      " 18  Notes             2211 non-null   object\n",
      " 19  AboutC            12470 non-null  object\n",
      " 20  Attach            1559 non-null   object\n",
      " 21  Year              19001 non-null  int64 \n",
      " 22  Month             19001 non-null  int64 \n",
      " 23  IT                19001 non-null  bool  \n",
      "dtypes: bool(1), int64(2), object(21)\n",
      "memory usage: 3.4+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c52f8df1-96c2-47e9-8d7b-1e0d177d71e5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
