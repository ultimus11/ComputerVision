
import re
sha = """
            Shah Rukh Khan (pronounced  ⓘ; born 2 November 1965), also known by the initialism SRK, is an Indian actor and film producer who works in Hindi films.
            Referred to in the media as the "Baadshah of Bollywood" and "King Khan,[a] he has appeared in more than 100 films, and earned numerous accolades, including 14 Filmfare Awards.
            He has been awarded the Padma Shri by the Government of India, as well as the Order of Arts and Letters and Legion of Honour by the Government of France.
            Khan has a significant following in Asia and the Indian diaspora worldwide.
            In terms of audience size and income, several media outlets have described him as one of the most successful film stars in the world.[b] Many of his films thematise Indian national identity and connections with diaspora communities, or gender, racial, social and religious differences and grievances."""

sha_lst = sha.split(" ")
sha_lst = [i.lower() for i in sha_lst]

unique_words = list(set(sha_lst))


count = 0
dict_counts_repetation_counts = {}
# count = sha_lst.count("khan")
for i in unique_words:
    count = 0
    # if "khan" in i:
    #     count+=1
    count = sha_lst.count(i)
    dict_counts_repetation_counts[i] = count

print(dict_counts_repetation_counts)
print(len(unique_words))
