from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, AutoModelWithLMHead
import spacy
from string import punctuation
from termcolor import colored

model_name = "deepset/roberta-base-squad2"
model_name_2 = "mrm8488/t5-base-finetuned-common_gen"
roberta_squad2 = pipeline('question-answering', model=model_name, tokenizer=model_name)
t5_finetuned = pipeline('text2text-generation', model=model_name_2, tokenizer=model_name_2)
spacy_sm = spacy.load("en_core_web_sm")

elvis = {
    "name": "Elvis Presley",
    "context": "Elvis Aaron Presley (January 8, 1935 – August 16, 1977), or simply Elvis, was an American singer and actor. Dubbed the 'King of Rock and Roll', he is regarded as one of the most significant cultural figures of the 20th century. His energized interpretations of songs and sexually provocative performance style, combined with a singularly potent mix of influences across color lines during a transformative era in race relations, led him to both great success and initial controversy. Presley was born in Tupelo, Mississippi, and relocated to Memphis, Tennessee, with his family when he was 13 years old. His music career began there in 1954, recording at Sun Records with producer Sam Phillips, who wanted to bring the sound of African-American music to a wider audience. Presley, on rhythm acoustic guitar, and accompanied by lead guitarist Scotty Moore and bassist Bill Black, was a pioneer of rockabilly, an uptempo, backbeat-driven fusion of country music and rhythm and blues. In 1955, drummer D. J. Fontana joined to complete the lineup of Presley's classic quartet and RCA Victor acquired his contract in a deal arranged by Colonel Tom Parker, who would manage him for more than two decades. Presley's first RCA Victor single, 'Heartbreak Hotel', was released in January 1956 and became a number-one hit in the United States. Within a year, RCA would sell ten million Presley singles. With a series of successful network television appearances and chart-topping records, Presley became the leading figure of the newly popular sound of rock and roll, though his performative style and promotion of the then-marginalized sound of African-Americans led to him being widely considered a threat to the moral well-being of the White American youth. In November 1956, Presley made his film debut in Love Me Tender. Drafted into military service in 1958, Presley relaunched his recording career two years later with some of his most commercially successful work. He held few concerts, however, and guided by Parker, proceeded to devote much of the 1960s to making Hollywood films and soundtrack albums, most of them critically derided. Some of his most famous films included Jailhouse Rock (1957), Blue Hawaii (1961), and Viva Las Vegas (1964). In 1968, following a seven-year break from live performances, he returned to the stage in the acclaimed television comeback special Elvis, which led to an extended Las Vegas concert residency and a string of highly profitable tours. In 1973, Presley gave the first concert by a solo artist to be broadcast around the world, Aloha from Hawaii. Years of prescription drug abuse and unhealthy eating habits severely compromised his health, and he died suddenly in 1977 at his Graceland estate at the age of 42. Having sold over 500 million records worldwide, Presley is recognized as the best-selling solo music artist of all time by Guinness World Records. He was commercially successful in many genres, including pop, country, rhythm & blues, adult contemporary, and gospel. Presley won three Grammy Awards, received the Grammy Lifetime Achievement Award at age 36, and has been inducted into multiple music halls of fame. He holds several records, including the most RIAA certified gold and platinum albums, the most albums charted on the Billboard 200, the most number-one albums by a solo artist on the UK Albums Chart, and the most number-one singles by any act on the UK Singles Chart. In 2018, Presley was posthumously awarded the Presidential Medal of Freedom. His favorite food was steak fajitas."
}

shakespeare = {
    "name": "William Shakespeare",
    "context": "William Shakespeare (bapt. 26 April 1564 – 23 April 1616) was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called England's national poet and the 'Bard of Avon' (or simply 'the Bard'). His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems, and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright. He remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted. Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna and twins Hamnet and Judith. Sometime between 1585 and 1592, he began a successful career in London as an actor, writer, and part-owner of a playing company called the Lord Chamberlain's Men, later known as the King's Men. At age 49 (around 1613), he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and whether the works attributed to him were written by others. Shakespeare produced most of his known works between 1589 and 1613. His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. He then wrote mainly tragedies until 1608, among them Hamlet, Romeo and Juliet, Othello, King Lear, and Macbeth, all considered to be among the finest works in the English language. In the last phase of his life, he wrote tragicomedies (also known as romances) and collaborated with other playwrights. Many of Shakespeare's plays were published in editions of varying quality and accuracy in his lifetime. However, in 1623, two fellow actors and friends of Shakespeare's, John Heminges and Henry Condell, published a more definitive text known as the First Folio, a posthumous collected edition of Shakespeare's dramatic works that included all but two of his plays. Its Preface was a prescient poem by Ben Jonson that hailed Shakespeare with the now famous epithet: 'not of an age, but for all time'."
}

print(colored("Welcome. Do you want to meet Elvis or Shakespeare?", 'red'))
answer = input()
selected = False

character = {"name": "", "context": ""}

while not selected:
    if answer == "Elvis":
        character = elvis
        selected = True
    elif answer == "Shakespeare":
        character = shakespeare
        selected = True
    else:
        print(colored("I did not understand your choice, please try again. Elvis or Shakespeare?", 'red'))
        answer = input()

# based on code in https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/
def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    doc = spacy_sm(text.lower()) 
    for token in doc:
        if token.text == "you":
            result.insert(0, "I")
        elif token.text == "your":
            result.insert(0, "my")
        else:
            if(token.text in spacy_sm.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)
    result = ' '.join(result).strip()
    return result

name = character["name"]
print(colored(f"Hello! I am {name}, what would you like to know about me? ", 'blue'))
user_input = input()
while(user_input != "quit()"):
    QA_input = {
        'question': user_input,
        'context': character["context"]
    }
    res = roberta_squad2(QA_input)
    if res['score'] < 0.009:
        print(colored("I cannot recall, it was a very long time ago. ", 'blue'))
    else:
        hotwords = get_hotwords(user_input)
        expanded_answer = f"{hotwords} {res['answer']}"
        res = t5_finetuned(expanded_answer)[0]
        print(colored(res['generated_text'], 'blue'))
    user_input = input()
