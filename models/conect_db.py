# !/usr/bin/env python
# -*- coding: utf-8 -*-
# def conect_database():
#     import mysql.connector as connector
#
#     # db = connector.connect(user='root',
#     #                        password='',
#     #                        host='127.0.0.1',
#     #                        database='new_dbzbot')
#
#     db = connector.connect(host="10.22.0.171",
#                          user="zbot",
#                          passwd="zbot@123",
#                          charset='utf8',
#                          use_unicode=True,
#                          db="zbot")
#     return db
#
#
# def get_train_data(bot_id):
#     db = conect_database()
#     cur = db.cursor()
#     cur.execute("SELECT samples.content AS samples_content, intents.intent FROM chatbot_intent JOIN intents ON intents.id = chatbot_intent.intentid LEFT JOIN samples ON samples.intent_id = intents.id WHERE chatbot_intent.botid ="+str(bot_id))
#     mydata = [list(x) for x in cur.fetchall()]
#     db.close()
#     return mydata
#
# # print(get_train_data(2))
#
# def get_answers(bot_id):
#     db = conect_database()
#     cur = db.cursor()
#     cur.execute("SELECT intent, answer FROM chatbot_intent JOIN intents ON intents.id = chatbot_intent.intentid LEFT JOIN intent_answers ON intent_answers.intent_id = intents.id WHERE chatbot_intent.botid = "+str(bot_id))
#     mydata = [list(x) for x in cur.fetchall()]
#     db.close()
#     return mydata
#
#
# def get_entities(bot_id):
#     db = conect_database()
#     cur = db.cursor()
#     cur.execute("SELECT entity, content FROM chatbot_entities JOIN entities ON entities.id = chatbot_entities.entityid JOIN entity_samples ON entity_samples.entity_id = entities.id WHERE chatbot_entities.botid = "+str(bot_id))
#     mydata = [list(x) for x in cur.fetchall()]
#     db.close()
#     return mydata
#
#
# def get_synonyms(bot_id):
#     db = conect_database()
#     cur = db.cursor()
#     cur.execute("SELECT entity_samples.content AS sample_content, synonym.content AS synonym_content FROM chatbot_entities JOIN entities ON entities.id = chatbot_entities.entityid JOIN entity_samples ON entity_samples.entity_id = entities.id LEFT JOIN synonym ON synonym.entity_sample_id = entity_samples.id WHERE chatbot_entities.botid ="+str(bot_id))
#     mydata = [list(x) for x in cur.fetchall()]
#     db.close()
#     # return mydata
#     return [['hanoi','thudo'],]
#
#
# def check_database(bot_id):
#     get_train_data(bot_id)
#     get_answers(bot_id)
#     get_entities(bot_id)
#     get_synonyms(bot_id)

# print(get_synonyms(2))
# print(get_entities(2))
# print(get_answers(2))
# print(get_train_data(2))

# def get_synonym_fromdb():
#     synonyms = get_synonyms()
#     return dict([[x[1], x[0]] for x in synonyms])

# synonyms = get_synonyms()
# my_list= []
# for synonym in synonyms:
#     my_list.append([synonym[1],synonym[0]])
# my_list = get_synonym_fromdb()
# print(my_list)
# token = u'Ha Noi'
# if token in my_list:
#     token = my_list.get(token)
#     print(token)# .lower()




def conect_database():
    import mysql.connector as connector

    db = connector.connect(unix_socket="/Applications/MAMP/tmp/mysql/mysql.sock", user='root',
                           password='root',
                           host='127.0.0.1',
                           database='dbzbot')

    # db = connector.connect(host="127.0.0.1",
    #                      user="root",
    #                      passwd="",
    #                      charset='utf8',
    #                      use_unicode=True,
    #                      db="chatbot")
    # db = connector.connect(host="10.22.0.171",
    #                      user="zbot",
    #                      passwd="zbot@123",
    #                      charset='utf8',
    #                      use_unicode=True,
    #                      db="zbot")
    return db


def get_train_data(bot_id, type_intent):
    db = conect_database()
    cur = db.cursor()
    cur.execute("SELECT samples.content AS samples_content, intents.intent FROM chatbot_intent JOIN intents ON intents.id = chatbot_intent.intentid LEFT JOIN samples ON samples.intent_id = intents.id WHERE (chatbot_intent.botid ="+str(bot_id)+str(" AND intents.context= ")+str(type_intent)+str(")"))
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata
# for i in get_train_data(2):
#     print(i[0])
def get_answers(bot_id):
    db = conect_database()
    cur = db.cursor()
    cur.execute("SELECT intent, answer FROM chatbot_intent JOIN intents ON intents.id = chatbot_intent.intentid LEFT JOIN intent_answers ON intent_answers.intent_id = intents.id WHERE chatbot_intent.botid = "+str(bot_id))
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_entities(bot_id):
    db = conect_database()
    cur = db.cursor()
    cur.execute("SELECT name, content FROM chatbot_entities JOIN enitty_categories ON enitty_categories.id = chatbot_entities.entityid JOIN entity_samples ON entity_samples.entity_category_id = enitty_categories.id WHERE chatbot_entities.botid = "+str(bot_id))
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata


def get_synonyms(bot_id):
    db = conect_database()
    cur = db.cursor()
    cur.execute("SELECT entity_samples.content AS sample_content, synonym.content AS synonym_content FROM chatbot_entities JOIN enitty_categories ON enitty_categories.id = chatbot_entities.entityid JOIN entity_samples ON entity_samples.entity_category_id = enitty_categories.id LEFT JOIN synonym ON synonym.entity_sample_id = entity_samples.id WHERE chatbot_entities.botid ="+str(bot_id))
    mydata = [list(x) for x in cur.fetchall()]
    db.close()
    return mydata
    # return [['hanoi','thudo'],]


def check_database(bot_id):
    get_train_data(bot_id)
    get_answers(bot_id)
    get_entities(bot_id)
    get_synonyms(bot_id)

# print(get_synonyms(16))
# print(get_entities(16))
# print(get_answers(16))
# print(get_train_data(16, 2))

# def get_synonym_fromdb():
#     synonyms = get_synonyms()
#     return dict([[x[1], x[0]] for x in synonyms])

# synonyms = get_synonyms()
# my_list= []
# for synonym in synonyms:
#     my_list.append([synonym[1],synonym[0]])
# my_list = get_synonym_fromdb()
# print(my_list)
# token = u'Ha Noi'
# if token in my_list:
#     token = my_list.get(token)
#     print(token)# .lower()
