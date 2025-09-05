diction = {'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'id': 'indonesian', 'ga': 'irish', 
'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian', 'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'or': 'odia', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 
'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu'}


lis = ['afr', 'amh', 'ara', 'asm', 'aze', 'aze_cyrl', 'bel', 'ben', 'bod', 'bos', 'bre', 'bul', 'cat', 'ceb', 
'ces', 'chi_sim', 'chi_sim_vert', 'chi_tra', 'chi_tra_vert', 'chr', 'cos', 'cym', 'dan', 'deu', 'div', 
'dzo', 'ell', 'eng', 'enm', 'epo', 'equ', 'est', 'eus', 'fao', 'fas', 'fil', 'fin', 'fra', 'frk', 'frm', 
'fry', 'gla', 'gle', 'glg', 'grc', 'guj', 'hat', 'heb', 'hin', 'hrv', 'hun', 'hye', 'iku', 'ind', 'isl', 
'ita', 'ita_old', 'jav', 'jpn', 'jpn_vert', 'kan', 'kat', 'kat_old', 'kaz', 'khm', 'kir', 'kmr', 'kor', 
'lao', 'lat', 'lav', 'lit', 'ltz', 'mal', 'mar', 'mkd', 'mlt', 'mon', 'mri', 'msa', 'mya', 'nep', 'nld',
'nor', 'oci', 'ori', 'osd', 'pan', 'pol', 'por', 'pus', 'que', 'ron', 'rus', 'san', 'sin', 'slk', 'slv',
'snd', 'spa', 'spa_old', 'sqi', 'srp', 'srp_latn', 'sun', 'swa', 'swe', 'syr', 'tam', 'tat', 'tel', 'tgk',
'tha', 'tir', 'ton', 'tur', 'uig', 'ukr', 'urd', 'uzb', 'uzb_cyrl', 'vie', 'yid', 'yor']

diction_as_list_shortcut = ['afr', 'alb', 'amh', 'ara', 'arm', 'aze', 'bas', 'bel', 'ben', 'bos', 'bul', 'cat', 'ceb', 'chi', 'chi', 'chi', 'cor', 'cro', 'cze', 'dan', 'dut', 'eng', 'esp', 'est', 'fil', 'fin', 'fre', 'fri', 'gal', 'geo', 'ger', 'gre', 'guj', 'hai', 'hau', 'haw', 'heb', 'heb', 'hin', 'hmo', 'hun', 'ice', 'igb', 'ind', 'iri', 'ita', 'jap', 'jav', 'kan', 'kaz', 'khm', 'kor', 'kur', 'kyr', 'lao', 'lat', 'lat', 'lit', 'lux', 'mac', 'mal', 'mal', 'mal', 'mal', 'mao', 'mar', 'mon', 'mya', 'nep', 'nor', 'odi', 'pas', 'per', 'pol', 'por', 'pun', 'rom', 'rus', 'sam', 'sco', 'ser', 'ses', 'sho', 'sin', 'sin', 'slo', 'slo', 'som', 'spa', 'sun', 'swa', 'swe', 'taj', 'tam', 'tel', 'tha', 'tur', 
'ukr', 'urd', 'uyg', 'uzb', 'vie', 'wel', 'xho', 'yid', 'yor', 'zul']
# for i,key in enumerate(diction):
    
#     d = diction[key][:3]
#     diction_as_list_shortcut.append(d)
# not_found = []
# for i in lis:
#     #if i in diction_as_list_shortcut:
#     if i[:3] in diction_as_list_shortcut:
#         not_found.append(i)

# print(len(lis)) # 124  googltrans  google translator
# print(len(diction)) # 107 pytesseract

# print("\n\n##########################################################\n\n")
# print(len(diction_as_list_shortcut)) # 107 pytesseract diction_as_list_shortcut
# # print(len(not_found))
# # print(not_found)
# print("\n\n##########################################################\n\n")
# print(len(not_found))
# print(not_found)

# start selnium
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import time

try:
    # language_code = "te"
    chrome_options = Options()
    #chrome_options.add_argument("--disable-extensions")
    #chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(executable_path="./chromedriver",options=chrome_options)
    driver.get('https://translate.google.com')
    whole_lang = driver.find_element(By.CSS_SELECTOR, "#yDmH0d > c-wiz > div > div.WFnNle > c-wiz > div.OlSOob > c-wiz > div.ccvoYb.EjH7wc > div.aCQag > c-wiz > div:nth-child(2) > c-wiz > div.ykTHSe > div > div.dykxn.MeCBDd.j33Gae > div > div:nth-child(2)")
    all = whole_lang.find_elements(By.CLASS_NAME, "qSb8Pe") # Llmcnf
    list_all_langs = []
    dic_all_lang = {}
    for i in all:
        name = i.find_element(By.CLASS_NAME, "Llmcnf").get_attribute("textContent")
        lang_code = i.get_attribute('data-language-code')

        list_all_langs.append(i.get_attribute('data-language-code'))
        dic_all_lang[lang_code] = name
    print(len(list_all_langs))
    print(dic_all_lang)
    # time.sleep(3)
    # driver.find_element_by_id("source").send_keys("how are you")
    # time.sleep(3)
    # output = driver.find_element_by_xpath("//span[@class='tlid-translation translation']//span").text
    # print(output)
    driver.quit()
except NoSuchElementException:
        print(NoSuchElementException.args)