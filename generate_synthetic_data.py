{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
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
   "execution_count": 10,
   "id": "2b00074a-ac11-467b-b323-91764643c299",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('C:/Users/ghass/hiring_bias_analysis/Datasets/data job posts.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
   "execution_count": 12,
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
   "execution_count": 13,
   "id": "9e72a91b-1645-4aa3-92b4-9248cabc7763",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               Year         Month\n",
      "count  19001.000000  19001.000000\n",
      "mean    2010.274722      6.493869\n",
      "std        3.315609      3.405503\n",
      "min     2004.000000      1.000000\n",
      "25%     2008.000000      3.000000\n",
      "50%     2011.000000      7.000000\n",
      "75%     2013.000000      9.000000\n",
      "max     2015.000000     12.000000\n",
      "jobpost\n",
      "Career Center NGO\\r\\nTITLE:  English Language Courses\\r\\nOPEN TO/ ELIGIBILITY CRITERIA:  Everyone\\r\\nLOCATION:  Yerevan, Armenia\\r\\nDETAIL DESCRIPTION:  Whether youre just getting started, already know\\r\\nEnglish and want to improve your skills, want to prepare for an exam or\\r\\ntest, you can find the right course here. \\r\\nCareer Center announces below mentioned English Language Courses:\\r\\nMAIN ENGLISH COURSE (consisting a total of 6 levels with the duration of\\r\\n3 months each):\\r\\n1. Beginner\\r\\n2. Elementary\\r\\n3. Pre-Intermediate\\r\\n4. Intermediate\\r\\n5. Upper-Intermediate\\r\\n6. Advanced (Final)\\r\\nSPECIAL COURSES (consisting a total of 3 levels with the duration of 3\\r\\nmonths each):\\r\\n- Business English - Pre-Intermediate\\r\\n- Business English - Intermediate\\r\\n- Business English - Upper-Intermediate (Final).\\r\\nBusiness English Courses also cover Special Business Writing and\\r\\nCommunication Classes.\\r\\nAPPLICATION PROCEDURES:  All interested candidates should visit Career\\r\\nCenter office and register as a member on Mondays - Fridays, from 9:00 -\\r\\n18:00. \\r\\nMonthly membership fee for all English language courses is 28,000 AMD.\\r\\nPlease note that the complete fee of any level (a total of 84,000 AMD)\\r\\nshould be paid at the time of starting the classes.\\r\\nRegistered students will pass a written placement test accompanied with\\r\\noral interview and be placed with a relevant group.\\r\\nRegistrations are not accepted by e-mail or telephone. For additional\\r\\ninquiries on registration or courses please contact us using below\\r\\ncontact information.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis training opportunity through Career Center and mention the URL of\\r\\nits website - www.careercenter.am, Thanks.\\r\\nAPPLICATION DEADLINE:  Rolling (Groups start their classes as soon as\\r\\nthere are 4-5 people).\\r\\nABOUT COMPANY:  Career Center NGO\\r\\nPhone/Fax: +(374 10) 560328\\r\\nE-mail: mailbox@... \\r\\nWeb site: www.careercenter.am \\r\\nAddress: 25 Abovyan Str., (next to the School named after Pushkin)\\r\\nYerevan, Armenia\\r\\nABOUT:  COURSES\\r\\n- Newly opened city central location;\\r\\n- Adequately furnished Dolby Digital classrooms with DVD, VCR and TV;\\r\\n- Specially designed ergonomic desks/ chairs;\\r\\n- 4-6 (max) people in a group ensuring efficiency of the courses;\\r\\n- Only highly qualified and certified language instructors selected by\\r\\nCareer Center will teach interested individuals with the latest methods\\r\\nusing the most decent study materials for each particular course.\\r\\n- Our classes are conducted in English language only.\\r\\n- Classes will take place in Career Center office, in a large, furnished\\r\\nand warm room.\\r\\n- For the whole duration of their studies students will be provided with\\r\\nnecessary books and materials, so they don't have to purchase or\\r\\nphotocopy any study materials. There are no additional charges for using\\r\\nthose materials. All provided textbooks must be returned to Career Center\\r\\nafter studies.\\r\\n- Sessions will be held 3 times a week and each of those will last 1.5\\r\\nhours.\\r\\n- Classes are on from 09:00 till 22:00, Monday-Saturdays. The attendance\\r\\nhours are assigned to each group according to their designated time\\r\\nschedule.\\r\\n- All students passing the final level course will get relevant\\r\\ncertificates upon completion of their course. Certificates will match to\\r\\nthe level of individual's knowledge determined by the final exam results.\\r\\nAttention: Those who fail to pass the final level exam test will not get\\r\\nany certificates!\\r\\nADDITIONAL NOTES:  When visiting our office for registration, please\\r\\nplan to spend about 30 minutes to take the language proficiency test.\\r\\nATTACHMENTS:\\r\\nThe following attachment(s) to this announcement can be downloaded from:http://www.careercenter.am/ccdspann.php?id=10123\\r\\n1. English Language Courses in Armenian - English_Courses_Armenian.doc\\r\\n(47K)\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.    11\n",
      "Career Center NGO\\r\\nTITLE:  English Language Courses\\r\\nOPEN TO/ ELIGIBILITY CRITERIA:  Everyone\\r\\nLOCATION:  Yerevan, Armenia\\r\\nDETAIL DESCRIPTION:  Whether youre just getting started, already know\\r\\nEnglish and want to improve your skills, want to prepare for an exam or\\r\\ntest, you can find the right course here. \\r\\nCareer Center announces below mentioned English Language Courses:\\r\\nGENERAL ENGLISH COURSE (consisting a total of 6 levels with the duration\\r\\nof 3 months each):\\r\\n1. Beginner\\r\\n2. Elementary\\r\\n3. Pre-Intermediate\\r\\n4. Intermediate\\r\\n5. Upper-Intermediate\\r\\n6. Advanced (Final)\\r\\nSPECIALIZED COURSES (consisting a total of 3 levels with the duration of\\r\\n3 months each):\\r\\n- Business English - Pre-Intermediate\\r\\n- Business English - Intermediate\\r\\n- Business English - Upper-Intermediate (Final).\\r\\nBusiness English Courses also cover Special Business Writing and\\r\\nCommunication Classes.\\r\\nAPPLICATION PROCEDURES:  All interested candidates should visit Career\\r\\nCenter office and register as a member on Mondays - Fridays, from 9:00 -\\r\\n17:30. \\r\\nMonthly membership fee for all English language courses is 28,000 AMD.\\r\\nPlease note that the complete fee of any level (a total of 84,000 AMD)\\r\\nshould be paid at the time of starting the classes.\\r\\nRegistered students will pass a written placement test accompanied with\\r\\noral interview and be placed with a relevant group.\\r\\nRegistrations are not accepted by e-mail or telephone. For additional\\r\\ninquiries on registration or courses please contact us using below\\r\\ncontact information.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis training opportunity through Career Center and mention the URL of\\r\\nits website - www.careercenter.am, Thanks.\\r\\nAPPLICATION DEADLINE:  Rolling (Groups start their classes as soon as\\r\\nthere are 4-6 people).\\r\\nABOUT COMPANY:  Career Center NGO\\r\\nPhone/Fax: +(374 10) 560328\\r\\nE-mail: mailbox@... \\r\\nWeb site: www.careercenter.am \\r\\nAddress: 25 Abovyan Str., (next to the School named after Pushkin)\\r\\nYerevan, Armenia\\r\\nABOUT:  COURSES\\r\\n- Newly opened city central location;\\r\\n- Adequately furnished Dolby Digital classrooms with DVD, VCR and TV;\\r\\n- Specially designed ergonomic desks/ chairs;\\r\\n- 4-6 (max) people in a group ensuring efficiency of the courses;\\r\\n- Only highly qualified and certified language instructors selected by\\r\\nCareer Center will teach interested individuals with the latest methods\\r\\nusing the most decent study materials for each particular course.\\r\\n- Our classes are conducted in English language only.\\r\\n- Classes will take place in Career Center office, in a furnished and\\r\\nwarm room.\\r\\n- For the whole duration of their studies students will be provided with\\r\\nnecessary books and materials, so they don't have to purchase or\\r\\nphotocopy any study materials. There are no additional charges for using\\r\\nthose materials.\\r\\n- Sessions will be held 3 times a week and each of those will last 1.5\\r\\nhours.\\r\\n- Classes are on from 09:00 till 22:00, Monday-Saturdays. The attendance\\r\\nhours are assigned to each group according to their designated time\\r\\nschedule.\\r\\n- All students passing the final level course will get relevant\\r\\ncertificates upon completion of their course. Certificates will match to\\r\\nthe level of individual's knowledge determined by the final exam results.\\r\\nAttention: Those who fail to pass the final level exam test will not get\\r\\nany certificates!\\r\\nADDITIONAL NOTES:  When visiting our office for registration, please plan\\r\\nto spend about 30 minutes to take the language proficiency test.\\r\\nATTACHMENTS:\\r\\nThe following attachment(s) to this announcement can be downloaded from:http://www.careercenter.am/ccdspann.php?id=11381\\r\\n1. English Language Courses in Armenian - English_Courses_Armenian.doc\\r\\n(42K)\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                9\n",
      "Career Center NGO\\r\\nTITLE:  Volunteer Registration & Request Process\\r\\nINTENDED AUDIENCE:  Respective organizations, Newly Graduates, Last year\\r\\nstudents and others\\r\\nLOCATION:  Yerevan, Armenia\\r\\nNEWS DETAILS:  Career Center is pleased to represent you its \"Volunteer\\r\\nCenter\" project. Within this project Career Center continuously solicits\\r\\napplications for free from volunteers and keeps an updated database of\\r\\nall individuals interested to work on volunteering bases. Meanwhile\\r\\nCareer Center accepts requests (applications) for volunteers from\\r\\ninterested organizations and in case of a match within our database we\\r\\ncreate a link with relevant candidates.\\r\\nThe main purpose of this project is to:\\r\\n1) Introduce the idea of volunteering in Armenia,\\r\\n2) Help organizations and communities to accomplish works which would\\r\\notherwise not be possible to make without volunteering input and \\r\\n3) Help individuals, especially newly graduates to gain relevant work\\r\\nexperience in their fields of specialization.\\r\\nThis project will help organizations to fill their volunteer openings in\\r\\na professional and timely manner.\\r\\nVOLUNTEER REGISTRATION PROCESS\\r\\nTo register as a volunteer please open the www.careerhouse.com website,\\r\\nregister as an Individual user (unless you have previously registered)\\r\\nand fill out your Resume. To make sure you are considered for\\r\\nvolunteering opportunities open the Availability section of the Resume\\r\\nand select the Yes option in the Willing to Volunteer field. \\r\\nVOLUNTEER REQUEST PROCESS\\r\\nIf you are looking for a volunteer/ employee please open\\r\\nwww.careerhouse.com website, register as an Organization (unless you\\r\\nhave previously registered), in the left side of the web page click\\r\\n\"Recruitment\", then click the \"Compose\" link, fill out, Preview and\\r\\nSubmit that form. \\r\\nOnce you do this, Career House professionals will start working on your\\r\\nrequest, and when already available will represent you with 3-5 potential\\r\\ncandidates, whom you will have a chance to interview and/or select the\\r\\none(s) that best match your requirements. \\r\\nGeneral Note \\r\\nTo view the Armenian version of the website and fill out the above\\r\\nmentioned forms in Armenian language, please open the www.careerhouse.am\\r\\nwebsite instead of .com . \\r\\nFor further inquiries about the Volunteer Center project, please feel\\r\\nfree to contact us using below contact information.\\r\\nABOUT COMPANY:  \\r\\nCareer Center - Promoting Equal Opportunities.\\r\\nPhone/Fax:  +(374 10) 560328 \\r\\nE-mail:     mailbox@... \\r\\nWeb site:   www.careercenter.am \\r\\nAddress:    25 Abovyan Str., \\r\\nYerevan, Armenia\\r\\nADDITIONAL NOTES:  Each organization can request one volunteer without\\r\\nsubscription fee. In order to request more volunteers, an organization\\r\\nshould consider to get a Career Center membership which is 22,500 AMD/\\r\\nmonth. The minimum acceptable membership duration is 3 months. The total\\r\\nnumber of volunteers an organization can request depends on the\\r\\nmembership months subscribed.\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    7\n",
      "Career Center NGO\\r\\nTITLE:  English Language Courses\\r\\nOPEN TO/ ELIGIBILITY CRITERIA:  Everyone\\r\\nLOCATION:  Yerevan, Armenia\\r\\nDETAIL DESCRIPTION:  Whether youre just getting started, already know\\r\\nEnglish and want to improve your skills, want to prepare for an exam or\\r\\ntest, you can find the right course here. \\r\\nCareer Center announces below mentioned English Language Courses:\\r\\nMAIN ENGLISH COURSE (consisting a total of 6 levels with the duration of\\r\\n3 months each):\\r\\n1. Beginner\\r\\n2. Elementary\\r\\n3. Pre-Intermediate\\r\\n4. Intermediate\\r\\n5. Upper-Intermediate\\r\\n6. Advanced (Final)\\r\\nSPECIAL COURSES (consisting a total of 3 levels with the duration of 3\\r\\nmonths each):\\r\\n- Business English - Pre-Intermediate\\r\\n- Business English - Intermediate\\r\\n- Business English - Upper-Intermediate (Final).\\r\\nBusiness English Courses also cover Special Business Writing and\\r\\nCommunication Classes.\\r\\nAPPLICATION PROCEDURES:  All interested candidates should visit Career\\r\\nCenter office and register as a member on Mondays - Fridays, from 9:00 -\\r\\n18:00. \\r\\nMonthly membership fee for all English language courses is 22,500 AMD.\\r\\nPlease note that the complete fee of any level (a total of 67,500 AMD)\\r\\nshould be paid at the time of starting the classes.\\r\\nRegistered students will pass a written placement test accompanied with\\r\\noral interview and be placed with a relevant group.\\r\\nRegistrations are not accepted by e-mail or telephone. For additional\\r\\ninquiries on registration or courses please contact us using below\\r\\ncontact information.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis training opportunity through Career Center and mention the URL of\\r\\nits website - www.careercenter.am, Thanks.\\r\\nAPPLICATION DEADLINE:  Rolling (Groups start their classes as soon as\\r\\nthere are 4-5 people).\\r\\nABOUT COMPANY:  Career Center NGO\\r\\nPhone/Fax: +(374 10) 560328\\r\\nE-mail: mailbox@... \\r\\nWeb site: www.careercenter.am \\r\\nAddress: 25 Abovyan Str., (next to the School named after Pushkin)\\r\\nYerevan, Armenia\\r\\nABOUT:  COURSES\\r\\n- Newly opened city central location;\\r\\n- Adequately furnished Dolby Digital classrooms with DVD, VCR and TV;\\r\\n- Specially designed ergonomic desks/ chairs;\\r\\n- 4-6 (max) people in a group ensuring efficiency of the courses;\\r\\n- Only highly qualified and certified language instructors selected by\\r\\nCareer Center will teach interested individuals with the latest methods\\r\\nusing the most decent study materials for each particular course.\\r\\n- Our classes are conducted in English language only.\\r\\n- Classes will take place in Career Center office, in a large, furnished\\r\\nand warm room.\\r\\n- For the whole duration of their studies students will be provided with\\r\\nnecessary books and materials, so they don't have to purchase or\\r\\nphotocopy any study materials. There are no additional charges for using\\r\\nthose materials. All provided textbooks must be returned to Career Center\\r\\nafter studies.\\r\\n- Sessions will be held 3 times a week and each of those will last 1.5\\r\\nhours.\\r\\n- Classes are on from 09:00 till 22:00, Monday-Saturdays. The attendance\\r\\nhours are assigned to each group according to their designated time\\r\\nschedule.\\r\\n- All students passing the final level course will get relevant\\r\\ncertificates upon completion of their course. Certificates will match to\\r\\nthe level of individual's knowledge determined by the final exam results.\\r\\nAttention: Those who fail to pass the final level exam test will not get\\r\\nany certificates!\\r\\nADDITIONAL NOTES:  When visiting our office for registration, please\\r\\nplan to spend about 30 minutes to take the language proficiency test.\\r\\nATTACHMENTS:\\r\\nThe following attachment(s) to this announcement can be downloaded from:http://www.careercenter.am/ccdspann.php?id=10123\\r\\n1. English Language Courses in Armenian - English_Courses_Armenian.doc\\r\\n(47K)\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.     7\n",
      "Career Center NGO\\r\\nTITLE:  English Language Courses\\r\\nOPEN TO/ ELIGIBILITY CRITERIA:  Everyone\\r\\nLOCATION:  Yerevan, Armenia\\r\\nDETAIL DESCRIPTION:  Whether youre just getting started, already know\\r\\nEnglish and want to improve your skills, want to prepare for an exam or\\r\\ntest, you can find the right course here. \\r\\nCareer Center announces below mentioned English Language Courses:\\r\\nGENERAL ENGLISH COURSE (consisting a total of 6 levels with the duration\\r\\nof 3 months each):\\r\\n1. Beginner\\r\\n2. Elementary\\r\\n3. Pre-Intermediate\\r\\n4. Intermediate\\r\\n5. Upper-Intermediate\\r\\n6. Advanced (Final)\\r\\nSPECIALIZED COURSES (consisting a total of 3 levels with the duration of\\r\\n3 months each):\\r\\n- Business English - Pre-Intermediate\\r\\n- Business English - Intermediate\\r\\n- Business English - Upper-Intermediate (Final).\\r\\nBusiness English Courses also cover Special Business Writing and\\r\\nCommunication Classes.\\r\\nAPPLICATION PROCEDURES:  All interested candidates should visit Career\\r\\nCenter office and register as a member on Mondays - Fridays, from 9:00 -\\r\\n17:30. \\r\\nMonthly membership fee for all English language courses is 28,000 AMD.\\r\\nPlease note that the complete fee of any level (a total of 84,000 AMD)\\r\\nshould be paid at the time of starting the classes.\\r\\nRegistered students will pass a written placement test accompanied with\\r\\noral interview and be placed with a relevant group.\\r\\nRegistrations are not accepted by e-mail or telephone. For additional\\r\\ninquiries on registration or courses please contact us using below\\r\\ncontact information.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis training opportunity through Career Center and mention the URL of\\r\\nits website - www.careercenter.am, Thanks.\\r\\nAPPLICATION DEADLINE:  Rolling (Groups start their classes as soon as\\r\\nthere are 4-6 people).\\r\\nABOUT COMPANY:  Career Center NGO\\r\\nPhone/Fax: +(374 10) 560328\\r\\nE-mail: mailbox@... \\r\\nWeb site: www.careercenter.am \\r\\nAddress: 25 Abovyan Str., (next to the School named after Pushkin)\\r\\nYerevan, Armenia\\r\\nABOUT:  COURSES\\r\\n- Newly opened city central location;\\r\\n- Adequately furnished Dolby Digital classrooms with DVD, VCR and TV;\\r\\n- Specially designed ergonomic desks/ chairs;\\r\\n- 4-6 (max) people in a group ensuring efficiency of the courses;\\r\\n- Only highly qualified and certified language instructors selected by\\r\\nCareer Center will teach interested individuals with the latest methods\\r\\nusing the most decent study materials for each particular course.\\r\\n- Our classes are conducted in English language only.\\r\\n- Classes will take place in Career Center office, in a furnished and\\r\\nwarm room.\\r\\n- For the whole duration of their studies students will be provided with\\r\\nnecessary books and materials, so they don't have to purchase or\\r\\nphotocopy any study materials. There are no additional charges for using\\r\\nthose materials.\\r\\n- Sessions will be held 3 times a week and each of those will last 1.5\\r\\nhours.\\r\\n- Classes are on from 09:00 till 22:00, Monday-Saturdays. The attendance\\r\\nhours are assigned to each group according to their designated time\\r\\nschedule.\\r\\n- All students passing the final level course will get relevant\\r\\ncertificates upon completion of their course. Certificates will match to\\r\\nthe level of individual's knowledge determined by the final exam results.\\r\\nAttention: Those who fail to pass the final level exam test will not get\\r\\nany certificates!\\r\\nADDITIONAL NOTES:  When visiting our office for registration, please plan\\r\\nto spend about 30 minutes to take the language proficiency test.\\r\\nATTACHMENTS:\\r\\nThe following attachment(s) to this announcement can be downloaded from:http://www.careercenter.am/ccdspann.php?id=10939\\r\\n1. English Language Courses in Armenian - English_Courses_Armenian.doc\\r\\n(46K)\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                7\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ..\n",
      "Ameriabank CJSC\\r\\nTITLE:  Financial Planning, Analysis and Methodology Division Senior\\r\\nSpecialist\\r\\nLOCATION:  Yerevan, Armenia\\r\\nJOB DESCRIPTION:  N/A\\r\\nJOB RESPONSIBILITIES:\\r\\n- Compile and make surveillance of Banks Budget, report on the Budget\\r\\nPerformance;\\r\\n- Introduce and periodically develop Banking Services Costing Methods,\\r\\ncalculate Cost Prices;\\r\\n- Record Banks incomes and costs by Business Units;\\r\\n- Participate in the elaboration of Banks strategic projects, providing\\r\\nbudgets and strategic planning;\\r\\n- Analyze Banks financial performance based on Balance sheet and Income\\r\\nstatement.\\r\\nREQUIRED QUALIFICATIONS:\\r\\n- University degree in Economics, Finance, Accounting or related fields;\\r\\nInternational accounting certificate is a plus;\\r\\n- Analytic and practical thinking;\\r\\n- Enthusiastic and creative;\\r\\n- Excellent knowledge of banking business and legislation of the RA;\\r\\n- Strong knowledge of English, Russian and Armenian languages;\\r\\n- Excellent knowledge of MS Office, Armenian Software for Banks;\\r\\n- At least 3 years of professional experience in banking, from which at\\r\\nleast 1.5 in the field of financial analysis.\\r\\nREMUNERATION/ SALARY:  Compensation: Varies from 100,000 to 2,000,000\\r\\nAMD as per Company grade S (Specialist).\\r\\nAPPLICATION PROCEDURES:  Please fill out the application form throughhttp://www.ameriabank.am/PDF/Ameriabank_Application_form.doc link or\\r\\nattached below, and together with CV, if applicable, send by e-mail at:hr.fin@... . In the subject line of your e-mail message please\\r\\nmention the title of the position you are applying for.\\r\\nNo personal visits, deliveries or phone calls, please.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis job opportunity through Career Center and mention the URL of its\\r\\nwebsite - www.careercenter.am, Thanks.\\r\\nOPENING DATE:  26 November 2008\\r\\nAPPLICATION DEADLINE:  12 December 2008\\r\\nATTACHMENTS:\\r\\nThe following attachment(s) to this announcement can be downloaded from:http://www.careercenter.am/ccdspann.php?id=8540\\r\\n1. Application form - Ameriabank_Appl_form.zip (69K)\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                1\n",
      "Armenbrok OJSC\\r\\nTITLE:  Credit Department Head\\r\\nSTART DATE/ TIME:  Immediately\\r\\nDURATION:  Long term\\r\\nLOCATION:  Yerevan, Armenia\\r\\nJOB DESCRIPTION:  Armenbrok OJSC is looking for a motivated, experienced\\r\\ncandidate for the position of the Head of its newly opened Credit\\r\\nOperations Division.\\r\\nJOB RESPONSIBILITIES:\\r\\n- Introduce efficient lending procedures;\\r\\n- Be responsible for credit portfolio and risk analysis, and\\r\\nrecommending corrective actions;\\r\\n- Establish, develop and update business relations with borrowers;\\r\\n- Perform routine monitoring of credit status, monitor use of proceeds;\\r\\n- Regulate, train and manage other staff members.\\r\\nREQUIRED QUALIFICATIONS:\\r\\n- University degree in economics or finance;\\r\\n- Fluent knowledge of Armenian, English and Russian languages;\\r\\n- At least 3 years of professional experience as credit specialist;\\r\\n- Ability to create and develop the credit division;\\r\\n- Ability to make decisions and ensure the results; \\r\\n- Strong analytical and global thinking skills;\\r\\n- Strong knowledge of legislation, CBA normative and requirements.\\r\\nREMUNERATION/ SALARY:  High\\r\\nAPPLICATION PROCEDURES:  If you meet the above requirements, please\\r\\nsubmit your CV to: hr@... mentioning\\r\\nthe position title you are applying for in the subject of your email.\\r\\nOnly short listed candidates will be invited to an interview.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis job opportunity through Career Center and mention the URL of its\\r\\nwebsite - www.careercenter.am, Thanks.\\r\\nOPENING DATE:  25 November 2008\\r\\nAPPLICATION DEADLINE:  15 December 2008\\r\\nABOUT COMPANY:  Armenbrok OJSC is an investment brokerage and consulting\\r\\ncompany in Armenia offering services to both local and foreign clients.\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                1\n",
      "Ameriabank CJSC\\r\\nTITLE:  Senior Software Developer\\r\\nLOCATION:  Yerevan, Armenia\\r\\nJOB DESCRIPTION:  N/A\\r\\nJOB RESPONSIBILITIES:\\r\\n- Design, develop, troubleshoot and debug software programs;\\r\\n- Exercise judgment within broadly defined practices and policies in\\r\\nselecting methods, techniques, and evaluation criteria for obtaining\\r\\nresults;\\r\\n- Design and implement sophisticated algorithms to solve complex\\r\\nproblems; \\r\\n- Coordinate system changes/installation with outsourced organizations;\\r\\n- Develop software applications for the bank;\\r\\n- Provide engineering of software related solutions for various\\r\\noperational needs;\\r\\n- Control the process of the development of new and the revision of\\r\\nalready existing functionality systems;\\r\\n- Develop and maintain installation and configuration procedures;\\r\\n- Repair and recover from software failures. Communicate with impacted\\r\\nconstituencies;\\r\\n- Configure/add new services as necessary;\\r\\n- Perform periodic performance and defect reporting to support capacity\\r\\nplanning.\\r\\nREQUIRED QUALIFICATIONS:\\r\\n- Higher education;\\r\\n- Excellent knowledge of VB; \\r\\n- Good knowledge of Java Script; \\r\\n- Experience of management of My SQL and MS SQL databases;\\r\\n- Good knowledge of AS3x, AS4x; COM technologies; WEB technologies; NET\\r\\ntechnologies;\\r\\n- Good knowledge of MS Windows XP/2003SF platform;\\r\\n- Ability to responsibly complete assigned tasks according to\\r\\ndeadlines;\\r\\n- Analytical thinking;\\r\\n- Sense of responsibility and accuracy;\\r\\n- Flexible and teamwork ability; \\r\\n- Fluency in Armenian and Russian languages; knowledge of technical and\\r\\nspoken English;\\r\\n- At least two years of relevant work experience.\\r\\nEthics: Unquestioned principles and behavior. Collaborative and\\r\\nresponsible work habits.\\r\\nREMUNERATION/ SALARY:  Varies from 100,000 to 2,000,000 AMD as per\\r\\nCompany grade S (Specialist).\\r\\nAPPLICATION PROCEDURES:  Please fill out the application form throughhttp://www.ameriabank.am/PDF/Ameriabank_Application_form.doc link or\\r\\nattached below, and together with CV, if applicable, send by e-mail at:hr.adm@... . In the subject line of your e-mail message please\\r\\nmention the title of the position you are applying for.\\r\\nNo personal visits, deliveries or phone calls, please.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis job opportunity through Career Center and mention the URL of its\\r\\nwebsite - www.careercenter.am, Thanks.\\r\\nOPENING DATE:  26 November 2008\\r\\nAPPLICATION DEADLINE:  12 December 2008\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       1\n",
      "Armenian Branch of SADE, JSC\\r\\nTITLE:  Receptionist/ Secretary\\r\\nOPEN TO/ ELIGIBILITY CRITERIA:  All eligible candidates\\r\\nSTART DATE/ TIME:  ASAP\\r\\nLOCATION:  Yerevan, Armenia\\r\\nJOB DESCRIPTION:  Armenian Branch of SADE, JSC is seeking a reliable\\r\\nprofessional for the position of Receptionist/ Secretary to perform\\r\\nroutine secretarial tasks including calendaring, receiving and screening\\r\\ntelephone calls, and reviewing incoming and outgoing correspondence; make\\r\\ntranslations from Armenian into French and vice versa.\\r\\nREQUIRED QUALIFICATIONS:\\r\\n- Proficiency in typing, MS Word, MS Excel, Outlook in a network\\r\\nenvironment;\\r\\n- Good organizational and communication skills, courteous personality\\r\\nwith moral behaviour;\\r\\n- Excellent knowledge of Armenian and French languages to make\\r\\ntranslations within both languages;\\r\\n- Knowledge of English language is a big plus;\\r\\n- At least 2 years of experience as receptionist;\\r\\n- Positive attitude, interpersonal skills.\\r\\nAPPLICATION PROCEDURES:  If you believe you meet the qualifications for\\r\\nthis position, please send your cover letter, resume and references to\\r\\nthe attention of Mr. Gevorg Gevorgyan at: g.gevorgyan@... andemploi_siege@... .\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis job opportunity through Career Center and mention the URL of its\\r\\nwebsite - www.careercenter.am, Thanks.\\r\\nOPENING DATE:  25 November 2008\\r\\nAPPLICATION DEADLINE:  05 December 2008\\r\\nABOUT COMPANY:  SADE is a French company specialized in hydraulic\\r\\nconstructions domain.\\r\\nADDITIONAL NOTES:  Work hours: Monday-Friday, 9:00-18:00 with one hour\\r\\nfor lunch.\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           1\n",
      "\"Kamurj\" UCO CJSC\\r\\n\\r\\n\\r\\nTITLE:  Lawyer in Legal Department\\r\\n\\r\\n\\r\\nTERM:  Full-time\\r\\n\\r\\n\\r\\nDURATION:  Indefinite\\r\\n\\r\\n\\r\\nLOCATION:  Yerevan, Armenia\\r\\n\\r\\n\\r\\nJOB DESCRIPTION:  \"Kamurj\" UCO CJSC is looking for a Lawyer in Legal\\r\\nDepartment. The incumbent will mainly be responsible for the\\r\\norganization's internal legal service, the development  of the internal\\r\\nregulation projects adopted by the board of the organization, supporting\\r\\nin the functions of the board as well as proper implementation of other\\r\\nfunctions related to the office work.\\r\\n\\r\\n\\r\\nJOB RESPONSIBILITIES:\\r\\n- Properly provide internal legal services of the organization;\\r\\n- Plan and develop internal regulation projects adopted by the Board;\\r\\napply legal expertise and provide conclusions;\\r\\n- Develop the organization's projects, internal and individual legal\\r\\nacts, project contracts and other documentation;\\r\\n- Provide legal consultation to the staff of the organization related to\\r\\ntheir functions;\\r\\n- Compile answers to the complaints of customers and partners;\\r\\n- Provide the advocacy of the company's interests in relation to the\\r\\ncomponents of other governmental authorities, individuals and legal\\r\\nentities;\\r\\n- Prepare and present reports.\\r\\n\\r\\n\\r\\nREQUIRED QUALIFICATIONS:\\r\\n- Higher legal education; Master's degree is a plus;\\r\\n- Existence of an advocate qualification is desirable;\\r\\n- At least 1 year of professional work experience in a related field;\\r\\n- Work experience in the RA financial system is desirable;\\r\\n- Excellent knowledge of the Armenian language; knowledge of the English\\r\\nlanguage is desirable;\\r\\n- Strong knowledge of MS Office, particularly Excel, Word and Outlook;\\r\\nwork experience with databases;\\r\\n- Ability to complete tasks on time and with proper quality;\\r\\n- Ability to carry out tasks during a short period of time;\\r\\n- Analytical and administrative skills;\\r\\n- Ambition to get results;\\r\\n- Ability to work in a high pressure environment;\\r\\n- Ability to find solutions in tense situations;\\r\\n- Communication and negotiation skills;\\r\\n- Ability to work in a team;\\r\\n- Communicative and punctual person with a high sense of responsibility.\\r\\n\\r\\n\\r\\nAPPLICATION PROCEDURES:  All qualified applicants are encouraged to\\r\\nsubmit their CVs in Armenian (compulsory) and English languages to:\\r\\nanahit.manukyan@... . Please clearly mention the position title in\\r\\nthe subject line of the e-mail. Or submit your CV at: 11 Kalents Str.,\\r\\nYerevan 0033, RA. Only short-listed candidates will be interviewed.\\r\\nPlease clearly mention in your application letter that you learned of\\r\\nthis job opportunity through Career Center and mention the URL of its\\r\\nwebsite - www.careercenter.am, Thanks.\\r\\n\\r\\n\\r\\nOPENING DATE:  30 December 2015\\r\\n\\r\\n\\r\\nAPPLICATION DEADLINE:  20 January 2016\\r\\n\\r\\n\\r\\nABOUT COMPANY:  \"Kamurj\" UCO CJSC is providing micro and small loans to\\r\\nlow-income families in urban and rural areas throughout Armenia. More\\r\\ninformation about \"Kamurj\" UCO CJSC is available at: www.kamurj.am.\\r\\n\\r\\n\\r\\n----------------------------------\\r\\nTo place a free posting for job or other career-related opportunities\\r\\navailable in your organization, just go to the www.careercenter.am\\r\\nwebsite and follow the \"Post an Announcement\" link.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       1\n",
      "Name: count, Length: 18892, dtype: int64\n",
      "jobpost                 0\n",
      "date                    0\n",
      "Title                  28\n",
      "Company                 7\n",
      "AnnouncementCode    17793\n",
      "Term                11325\n",
      "Eligibility         14071\n",
      "Audience            18361\n",
      "StartDate            9326\n",
      "Duration             8203\n",
      "Location               32\n",
      "JobDescription       3892\n",
      "JobRequirment        2522\n",
      "RequiredQual          484\n",
      "Salary               9379\n",
      "ApplicationP           60\n",
      "OpeningDate           706\n",
      "Deadline               65\n",
      "Notes               16790\n",
      "AboutC               6531\n",
      "Attach              17442\n",
      "Year                    0\n",
      "Month                   0\n",
      "IT                      0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Summary statistics for numerical data\n",
    "print(df.describe())\n",
    "\n",
    "# Summary for categorical data\n",
    "print(df['jobpost'].value_counts())\n",
    "\n",
    "# Check for missing values\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ced82e55-4741-4d81-bf96-2a83a50c8f0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing values for a specific column\n",
    "df['jobpost'].fillna(df['jobpost'].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1cb6c153-2bb5-499d-b84f-4daf2378a482",
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop rows with missing values\n",
    "df.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "4d00e352-f99c-4b72-b9a2-22407d6a73fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting faker\n",
      "  Downloading Faker-30.1.0-py3-none-any.whl.metadata (15 kB)\n",
      "Requirement already satisfied: python-dateutil>=2.4 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from faker) (2.9.0.post0)\n",
      "Requirement already satisfied: typing-extensions in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from faker) (4.11.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from python-dateutil>=2.4->faker) (1.16.0)\n",
      "Downloading Faker-30.1.0-py3-none-any.whl (1.8 MB)\n",
      "   ---------------------------------------- 0.0/1.8 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.0/1.8 MB ? eta -:--:--\n",
      "   - -------------------------------------- 0.1/1.8 MB 975.2 kB/s eta 0:00:02\n",
      "   --------- ------------------------------ 0.5/1.8 MB 4.0 MB/s eta 0:00:01\n",
      "   ------------------- -------------------- 0.9/1.8 MB 5.5 MB/s eta 0:00:01\n",
      "   -------------------- ------------------- 0.9/1.8 MB 4.6 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 1.8/1.8 MB 7.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 1.8/1.8 MB 6.9 MB/s eta 0:00:00\n",
      "Installing collected packages: faker\n",
      "Successfully installed faker-30.1.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install faker\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "8f12245d-2dd7-42cc-974c-19854eb96431",
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
      "  OpeningDate                                       Deadline Notes  \\\n",
      "0         NaN                                26 January 2004   NaN   \n",
      "1         NaN                                12 January 2004   NaN   \n",
      "2         NaN  20 January 2004\\r\\nSTART DATE:  February 2004   NaN   \n",
      "3         NaN      23 January 2004\\r\\nSTART DATE:  Immediate   NaN   \n",
      "4         NaN                         20 January 2004, 18:00   NaN   \n",
      "\n",
      "                                              AboutC Attach  Year Month  \\\n",
      "0                                                NaN    NaN  2004     1   \n",
      "1  The International Research & Exchanges Board (...    NaN  2004     1   \n",
      "2  The Caucasus Environmental NGO Network is a\\r\\...    NaN  2004     1   \n",
      "3                                                NaN    NaN  2004     1   \n",
      "4                                                NaN    NaN  2004     1   \n",
      "\n",
      "      IT  gender   race  \n",
      "0  False    male  white  \n",
      "1  False  female  asian  \n",
      "2  False  female  asian  \n",
      "3  False  female  white  \n",
      "4   True    male  white  \n",
      "\n",
      "[5 rows x 26 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import random\n",
    "from faker import Faker\n",
    "\n",
    "fake = Faker()\n",
    "\n",
    "# Loading existing dataset\n",
    "df = pd.read_csv('C:/Users/ghass/hiring_bias_analysis/Datasets/data job posts.csv')\n",
    "\n",
    "# Function to create synthetic demographics\n",
    "def generate_demographics(n):\n",
    "    genders = ['male', 'female']\n",
    "    races = ['white', 'black', 'asian', 'hispanic', 'other']\n",
    "    \n",
    "    synthetic_data = {\n",
    "        'gender': random.choices(genders, k=n),\n",
    "        'race': random.choices(races, weights=[0.6, 0.15, 0.15, 0.05, 0.05], k=n)\n",
    "    }\n",
    "    return pd.DataFrame(synthetic_data)\n",
    "\n",
    "# Generate synthetic data for the number of rows in your dataset\n",
    "synthetic_df = generate_demographics(len(df))\n",
    "\n",
    "# Merge synthetic data with the original dataset\n",
    "df = pd.concat([df, synthetic_df], axis=1)\n",
    "\n",
    "# Save the updated dataset\n",
    "df.to_csv('updated_dataset.csv', index=False)\n",
    "\n",
    "print(df.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "565ef385-e63c-4178-a6fb-857f47b2d02d",
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
