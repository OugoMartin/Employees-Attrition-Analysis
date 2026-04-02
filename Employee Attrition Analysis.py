#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[7]:


df=pd.read_excel('Employee_Cleaned.xls')


# In[8]:


print(df.head())


# In[9]:


plt.figure(figsize=(7,5))
df["LeaveOrNot"].value_counts().sort_index().plot(kind="bar")
plt.title("Employee Attrition Count")
plt.xlabel("LeaveOrNot (0 = Stayed, 1 = Left)")
plt.ylabel("Employee Count")
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
(df.groupby("City")["LeaveOrNot"].mean().mul(100).sort_values(ascending=False).plot(kind="bar"))
plt.title("Attrition Rate by City")
plt.xlabel("City")
plt.ylabel("Attrition Rate (%)")
plt.show()


# In[19]:


plt.figure(figsize=(8,5))

df.groupby("PaymentTier")["LeaveOrNot"].mean().mul(100).sort_index().plot(kind="bar")

plt.title("Attrition Rate by Payment Tier")
plt.xlabel("Payment Tier")
plt.ylabel("Attrition Rate (%)")

plt.show()


# In[21]:


plt.figure(figsize=(8,5))

plt.hist(df.loc[df["LeaveOrNot"] == 0, "Age"], bins=15, alpha=0.7, label="Stayed")
plt.hist(df.loc[df["LeaveOrNot"] == 1, "Age"], bins=15, alpha=0.7, label="Left")

plt.title("Age Distribution by Attrition")
plt.xlabel("Age")
plt.ylabel("Employee Count")

plt.legend()
plt.show()


# In[24]:


encoded = pd.get_dummies(df, drop_first=True)

encoded.head()


# In[25]:


from sklearn.model_selection import train_test_split

X = encoded.drop(columns=["LeaveOrNot"])
y = encoded["LeaveOrNot"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)


# In[30]:


feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top = feature_importance.head(15).iloc[::-1]

plt.figure(figsize=(10,6))
top.plot(kind="barh")

plt.title("Top 15 Important Features for Attrition")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()


# In[33]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

