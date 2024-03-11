###############################################################################
#--------------------------- IMPORTING LIBRARIES -----------------------------------------------#
###############################################################################
library(seededlda)
library(quanteda)
library(quanteda.textmodels)
library(quanteda.sentiment)
library(readtext)
library(caret)
library(dplyr)
###############################################################################
#-----------------------------------------------------------------------------#
###############################################################################

###############################################################################
#------------------------Naive Bayes Classifier-----------------------------#
###############################################################################

jobs <- readtext("fake_job_postings.csv", text_field = "description", docid_field = "job_id")


## Define corpus
jobs_corpus <- corpus(jobs)
summary(jobs_corpus, 10)

## Tokenize
jobs_tokens <- tokens(jobs_corpus, remove_punct = TRUE, remove_numbers = TRUE,
                     remove_url = TRUE, remove_symbols = TRUE, remove_separators = TRUE)
jobs_tokens <- tokens_tolower(jobs_tokens)
jobs_tokens <- tokens_remove(jobs_tokens, stopwords("english"))
jobs_tokens <- tokens_wordstem(jobs_tokens)
jobs_tokens

## DFM
jobs_dfm <- dfm(jobs_tokens)
jobs_dfm

##TRAIN & TEST SETS
set.seed(0007)
train_rows <- createDataPartition(docvars(jobs_corpus, "fraudulent"), p = 0.8,
                                  list = FALSE)
jobs_dfm_train <- dfm_subset(jobs_dfm, docid(jobs_dfm) %in% train_rows)
jobs_dfm_test <- dfm_subset(jobs_dfm, !(docid(jobs_dfm) %in% train_rows))

jobs_dfm_train
jobs_dfm_test

## NB Model
nb_model <- textmodel_nb(jobs_dfm_train, docvars(jobs_dfm_train, "fraudulent"))
summary(nb_model)

jobs_dfm_matched <- dfm_match(jobs_dfm_test, featnames(jobs_dfm_train))
docvars(jobs_dfm_test, "fraud_nb_pred") <- predict(nb_model, newdata = jobs_dfm_matched)
docvars(jobs_dfm_test, "fraud_nb_pred") <- factor(docvars(jobs_dfm_test, "fraud_nb_pred"))
docvars(jobs_dfm_test, "fraudulent") <- factor(docvars(jobs_dfm_test, "fraudulent"))

confusionMatrix(docvars(jobs_dfm_test, "fraud_nb_pred"),
                reference = docvars(jobs_dfm_test, "fraudulent"),
                positive = "1")


###############################################################################
#--------------------------- LOGISTIC REGRESSION -----------------------------------------------#
###############################################################################
set.seed(12345)

jobs_LR <- read.csv("fake_job_postings.csv")
jobs_LR<- transform(jobs_LR, Real_Fake = ifelse(fraudulent== 1, "Fake", "Real"))
jobs_LR <- transform(jobs_LR, Real_Fake = factor(Real_Fake))
jobs_LR <- transform(jobs_LR, Real_Fake = relevel(Real_Fake, ref = "Real"))
summary(jobs_LR$Real_Fake)

##### TRYING TO MAKE A RECODE:
#SALARY VARIABLE:
#jobs_LR <- transform(jobs_LR, salary = ifelse(is.na(jobs_LR$salary_range) %in% NA, "0",  "1"))
#jobs_LR$salary<- ifelse(jobs_LR$salary_range %in% c(is.na(jobs_LR$salary_range)),1,0)
#jobs_LR  <- transform(jobs_LR , salary = case_when(is.na(salary_range) == TRUE ~ "1",
                                                  # is.na(salary_range) == FALSE ~ "0"))
#jobs_LR  <- transform(jobs_LR , salary = jobs_LR[is.na(jobs_LR$salary_range)] <- 1)
#jobs_LR  <- transform(jobs_LR , salary = jobs_LR[!is.na(jobs_LR$salary_range)] <- 0)
#write.csv(jobs_LR, "jobs.csv")

#Recoding:
#SALARY & LOGO VARIABLES:
jobs_LR <- transform(jobs_LR, salary = ifelse(jobs_LR$salary_range == "", "0",  "1"))
jobs_LR <- transform(jobs_LR, company_logo = ifelse(jobs_LR$has_company_logo == 0, "0",  "1"))
write.csv(jobs_LR, "jobs.csv")

#Factor & relevel education variable:
factor_education <- factor(jobs_LR$required_education)
factor_education<- relevel(factor_education, ref = "")
levels(factor(jobs_LR$required_education))

#EDUCATION VARIABLE:
#jobs_LR  <- transform(jobs_LR, required_education_categories = case_when(required_education == "Associate Degree" ~ 0,required_education == "Bachelor's Degree" ~ 1, required_education == "Certification" ~ 2, required_education == "Doctorate" ~ 3, required_education == "High School or equivalent" ~ 4,
#required_education == "Master's Degree" ~ 5, required_education == "Professional" ~ 6,required_education == "Some College Coursework Completed" ~ 7,
#required_education == "Some High School Coursework" ~ 8, required_education == "Unspecified" ~ 9,
#required_education == "Vocational" ~ 10, required_education == "Vocational - Degree" ~ 11,
#required_education == "Vocational - HS Diploma" ~ 12))
#jobs_LR  <- mutate_at(jobs_LR , c("required_education_categories"), ~replace(., is.na(.), 999))
#write.csv(jobs_LR, "jobs_for_LR.csv")

# TRAIN & TEST SETS:
train_rows <- createDataPartition(jobs_LR$fraudulent, p = 0.8, list = FALSE)
jobs_LR_train <- jobs_LR[train_rows, ]
jobs_LR_test <- jobs_LR[-train_rows, ]
str(jobs_LR_train)
str(jobs_LR_test)


#model_logit <- glm(Real_Fake ~ has_company_logo + required_education_categories + salary,data = jobs_LR_train,family = "binomial")

model_logit <- glm(Real_Fake ~ has_company_logo + required_education + salary,data = jobs_LR_train,family = "binomial")
summary(model_logit)
coef(model_logit)
exp(coef(model_logit))


predict_probability <- predict(object = model_logit,
                               newdata = jobs_LR_test, 
                               type = "response")
head(predict_probability)

predict_fake <- ifelse(predict_probability > 0.16,
                         yes = "Fake",
                         no = "Real")

predict_fake <- factor(predict_fake)
predict_fake <- relevel(predict_fake, ref = "Real")
summary(predict_fake)

confusionMatrix(data = predict_fake,
                reference = jobs_LR_test$Real_Fake, # note this uses the test set
                positive = "Fake")


###############################################################################
#----------------------------- Topic Modeling -----------------------------------------------#
###############################################################################

set.seed(00077)

#reading:
jobs2 <- readtext("fake_job_postings.csv", text_field = "description", docid_field = "job_id")
jobs2

#pre-processing levels:
jobs2_corpus <- corpus(jobs2)
jobs2_tokens <- tokens(jobs2_corpus)
jobs2_tokens_clean <- tokens(jobs2_corpus, remove_punct = TRUE,
                             remove_symbols = TRUE, remove_numbers = TRUE,
                             remove_separators = TRUE,  remove_twitter = TRUE,
                             remove_hyphens = FALSE,
                             remove_url = TRUE,)

jobs2_tokens_clean <- tokens_tolower(jobs2_tokens_clean)
jobs2_dfm <- dfm(jobs2_tokens_clean)
jobs2_dfm_clean <- dfm_remove(jobs2_dfm, stopwords("english"))
extra_stop <- c("us","nbsp","amp", "re", "work", "can", "job","¿\\u05bf", " \\u05bf","\\u05bf", "\\u05b2", "¢\\u05b2", "·\\u05b2", ":\\u05b2", "*\\u05b2", "s", "â")
jobs2_dfm_clean <- dfm_remove(jobs2_dfm_clean, extra_stop)
jobs2_dfm_clean

################## LDA ALL DATA !!!!! ##############################

#jobs2_lda <- textmodel_lda(jobs2_dfm_clean, k = 4)
#jobs2_lda

#jobs2_topic_terms <- terms(jobs2_lda)
#jobs2_topic_terms
#write.csv(jobs2_topic_terms, "top10_terms_ALL_jobs.csv")

#jobs2_topics <- c("Temptation/Extra Benefit", "Fields of Work/Industries", "Areas of Responsibility", "Terms of Employment")

#jobs2_theta <- jobs2_lda$theta
#colnames(jobs2_theta) <- jobs2_topics
#head(jobs2_theta)
#colMeans(jobs2_theta)
#jobs2_doc_topics <- topics(jobs2_lda)

#jobs2_doc_topics  <- case_when(jobs2_doc_topics  == "topic1" ~ jobs2_topics[1],
#                               jobs2_doc_topics == "topic2" ~ jobs2_topics[2],
#                               jobs2_doc_topics == "topic3" ~ jobs2_topics[3],
#                               jobs2_doc_topics == "topic4" ~ jobs2_topics[4])

#write.csv(jobs2_doc_topics, "LDA_full_data_result.csv")


################# LDA ONLY ON FAKE JOBS #####################
only_fraud_dfm <- dfm_subset(jobs2_dfm_clean, fraudulent == 1)
only_fraud_lda <- textmodel_lda(only_fraud_dfm, k = 4)
only_fraud_topic_terms <- terms(only_fraud_lda)
only_fraud_topic_terms
write.csv(only_fraud_topic_terms, "NEW_top10_terms_FRAUD_jobs.csv")

fraud_jobs <- subset(jobs2, fraudulent == 1)
fraud_jobs <- transform(fraud_jobs, topic = topics(only_fraud_lda))
head(fraud_jobs)
fraud_jobs <- transform(fraud_jobs, topic = topics(only_fraud_lda))
write.csv(fraud_jobs, "NEW_fraud_jobs_LDA_results.csv")

# topics names:
jobs2_topics <- c("Requirements",	"Temptation/Extra Benefit",	"Terms of Employment",	"Areas of Responsibility/Industries")

#extract theta and popular topic:
jobs2_theta <- only_fraud_lda$theta
colnames(jobs2_theta) <- jobs2_topics
head(jobs2_theta)
colMeans(jobs2_theta)

#extract topics
jobs2_doc_topics <- topics(only_fraud_lda)
head(jobs2_doc_topics)
#make a recode to english topics
jobs2_topics_english <- case_when(jobs2_doc_topics  == "topic1" ~ jobs2_topics[1],
                                  jobs2_doc_topics  == "topic2" ~ jobs2_topics[2],
                                  jobs2_doc_topics  == "topic3" ~ jobs2_topics[3],
                                  jobs2_doc_topics  == "topic4" ~ jobs2_topics[4])
# write it to a new file:
head(jobs2_topics_english)
fraud_jobs <- subset(jobs2, fraudulent == 1)
fraud_jobs <- transform(fraud_jobs, topic = jobs2_topics_english)
write.csv(fraud_jobs, "LDA_only_FRAUD_data_result.csv")


###############################################################################
#--------------------------- WORD CLOUD -----------------------------------------------#
###############################################################################
library(quanteda.textplots)
library(viridis)
textplot_wordcloud(only_fraud_dfm, min_count = 30, color = viridis(6),
                   rotation = 0.25, min_size = 1, max_size = 6)