"""Decode an Indian registration plate into the state and district that issued it.

WHAT THIS IS
------------
An Indian plate reads like ``PB39N0203``. The first two letters are the
state or union territory. The digits after them are the RTO district code
within that state. So ``PB39`` is Rajpura in Punjab, and ``HR26`` is
Gurgaon in Haryana. Everything after that is the vehicle's own series and
number, and says nothing about geography.

This module does that lookup and nothing else. It runs no models, touches
no video, and has no dependencies beyond the standard library.

WHY IT DECODES ON READ AND NOT AT CAPTURE
-----------------------------------------
The mapping is a pure function of the plate string, so there is no reason
to freeze its answer into the database at the moment of capture. Decoding
when the record is served means that correcting or extending the table
below also corrects every plate already recorded, instead of leaving old
rows carrying an old answer. It also keeps the ANPR pipeline untouched.

HONESTY ABOUT COVERAGE
----------------------
The state table is complete: all 36 states and union territories, plus the
BH (Bharat) series, which is deliberately nationwide and has no district.

The district table is NOT complete, and this module never pretends
otherwise. India has well over a thousand RTO codes, they change as
districts are created and split, and the public list used here was
compiled before Telangana (2014) and Ladakh (2019) existed. So:

  * a code that is in the table returns its district;
  * a code that is not returns ``None`` for the district, and the caller
    is expected to show the state alone rather than invent a district.

``decode()`` reports which of those happened in ``district_known``, so the
interface can say "district not in table" instead of showing a blank that
looks like a failed read.

KNOWN LIMITS, STATED RATHER THAN HIDDEN
---------------------------------------
* Telangana (TS/TG) plates decode to the state but have no districts here,
  because the source list predates the 2014 split. Hyderabad-area codes
  appear under Andhra Pradesh for the same reason.
* Ladakh (LA) decodes to the state only; its codes were carved out of
  Jammu & Kashmir in 2019 and are not in the source list.
* Odisha appears as both OD (current) and OR (pre-2011). Uttarakhand
  appears as both UK (current) and UA (pre-2007). Both spellings decode.
* Some codes in the source are marked "not in use", "not allotted" or
  belong to a government or transport fleet rather than a place. They are
  kept verbatim rather than cleaned up, because a plate really can carry
  one and the honest answer is what the register says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

__all__ = ["RtoOrigin", "decode", "state_of", "district_of", "coverage"]


# ---------------------------------------------------------------------------
# States and union territories
# ---------------------------------------------------------------------------
# Complete. The two-letter prefix is the only part of a plate that is
# guaranteed to carry geography.

STATES: dict[str, str] = {
    "AN": "Andaman & Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman & Diu",
    "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "TG": "Telangana",
    "UA": "Uttarakhand",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    # The Bharat series is issued nationally to vehicles that move between
    # states. It has no district by design, which is not a gap in the data.
    "BH": "Bharat series (all-India)",
}

#: Prefixes that are nationwide rather than regional.
NATIONWIDE = {"BH"}


# ---------------------------------------------------------------------------
# District codes
# ---------------------------------------------------------------------------
# Stored as one "CODE|District" per line rather than as a dict literal: it is
# data, it is long, and a flat block is far easier to check against the
# published register than several hundred lines of Python punctuation.
#
# Compiled from the public India-Codes RTO register. Incomplete by nature -
# see the module docstring. A code that is absent returns None, and the
# caller shows the state alone.

_DISTRICT_TABLE = """
AN01|Port Blair, Andaman District
AN02|Car Nicobar, Nicobar District
AP01|Adilabad / Mancherial / Nirmal
AP02|Anantapur / Hindupur
AP03|Chittoor / Tirupati / Madanapalle / Rajampet
AP04|Kadapa / Proddatur
AP05|Kakinada / Amalapuram / Rajahmundry
AP06|Kakinada
AP07|Guntur / Piduguralla / Narasaraopet / Repalle
AP08|Guntur
AP09|Khairtabad, Hyderabad District
AP10|Secunderabad, Hyderabad District
AP11|Malakpet, Hyderabad District
AP12|Bahadurpura, Hyderabad District
AP13|Mehdipatnam, Hyderabad District
AP14|Karimnagar
AP15|Karimnagar / Jagtial / Peddapalli
AP16|Bejawada / Gudivada / Machilipatnam / Nandigama
AP17|Bejawada / Gudivada / Machilipatnam / Nandigama
AP18|Nuzvid
AP19|Nuzvid
AP20|Khammam / Kothagudem / Sathupally / Yellandu
AP21|Kurnool / Adoni / Nandyal
AP22|Mahaboobnagar / Pebbair / Gadwal
AP23|Medak / Siddipet / Sangareddy
AP24|Nalgonda / Bhongir / Suryapet
AP25|Nizamabad / Kamareddy
AP26|Nellore / Kavali / Gudur / Sullurpeta
AP27|Ongole / Chirala / Markapur / Prakasam
AP28|Rangareddy
AP29|Rangareddy
AP30|Srikakulam / Tekkali / Palasa / Sompeta / Ichchapuram / Kaviti
AP31|Visakhapatnam / Gajuwaka / Anakapalli
AP32|Visakhapatnam
AP33|Gajuwaka
AP34|Gajuwaka
AP35|Vizianagaram
AP36|Warangal / Jangaon / Mahabubabad
AP37|Eluru / Bhimavaram / Tadepalligudem / Tanuku / Palakollu
AR01|Itanagar
AR02|Itanagar (state transport commercial vehicles)
AR03|Tawang
AR04|Bomdila
AR05|Seppa
AR06|Ziro
AR07|Daporijo
AR08|Along
AR09|Pasighat
AR10|Anini
AR11|Tezu
AR12|Changlang
AR13|Khonsa
AR14|Yingkiong
AS01|Guwahati, Kamrup
AS02|Nagaon
AS03|Jorhat
AS04|Sibsagar
AS05|Golaghat
AS06|Dibrugarh
AS07|Lakhimpur
AS08|Dima Hasao
AS09|Karbi Anglong
AS10|Karimganj
AS11|Cachar
AS12|Sonitpur
AS13|Darrang
AS14|Nalbari
AS15|Barpeta
AS16|Kokrajhar
AS17|Dhubri
AS18|Goalpara
AS19|Bongaigaon
AS20|Assam State Transport Corporation (ASTC)
AS21|Marigaon
AS22|Dhemaji
AS23|Tinsukia
AS24|Hailakandi
AS25|Guwahati (no longer used)
AS27|Udalguri
AS30|Assam Police vehicles
AS31|Assam Police
BR01|Patna
BR02|Gaya
BR03|Bhojpur
BR04|Chapra
BR05|Motihari
BR06|Muzaffarpur
BR07|Darbhanga
BR08|Munger
BR09|Begusarai
BR10|Bhagalpur
BR11|Purnea
BR18|Chaibasa
BR19|Saharsa
BR21|Nalanda
BR22|Bettiah
BR23|Giridih
BR24|Dehri
BR25|Jehanabad
BR26|Aurangabad
BR27|Nawada
BR28|Gopalganj
BR29|Siwan
BR30|Sitamarhi
BR31|Vaishali
BR32|Madhubani
BR33|Samastipur
BR34|Khagaria
BR37|Kishanganj
BR38|Araria
BR39|Katihar
BR43|Madhepura
BR44|Buxar
BR45|Bhabua
BR46|Jamui
BR47|Koderma
BR50|Supaul
BR51|Banka
BR52|Sheikhpura
BR53|Lakhisarai
CG01|Governor of Chhattisgarh
CG02|Government of Chhattisgarh
CG03|Chhattisgarh Police
CG04|Raipur
CG05|Dhamtari
CG06|Mahasamund
CG07|Durg
CG08|Rajnandgaon
CG09|Kawardha
CG10|Bilaspur
CG11|Janjgir-Champa
CG12|Korba
CG13|Raigarh
CG14|Jashpur
CG15|Ambikapur
CG16|Baikunthpur
CG17|Jagdalpur
CG18|Dantewada
CG19|Kanker
CG20|Bijapur
CH01|Chandigarh
CH02|Chandigarh (commercial vehicles)
CH03|Chandigarh
CH04|Chandigarh
DD02|Diu
DD03|Daman
DL1|Delhi North: Mall Road
DL2|New Delhi: Tilak Marg
DL3|South Delhi: Sheikh Sarai
DL4|West Delhi I: Janakpuri
DL5|North East Delhi: Loni Road
DL6|Central Delhi: Sarai Kale Khan
DL7|East Delhi I: Mayur Vihar
DL8|North West Delhi I: Wazirpur
DL9|South West Delhi I: Janakpuri / Palam
DL10|West Delhi II: Raja Garden
DL11|North West Delhi II: Rohini
DL12|South West Delhi II: Vasant Vihar
DL13|East Delhi II: Surajmal Vihar / Shahdara
DL14|Sonipat (NCR, situated in Haryana)
DL15|Gurgaon (NCR, situated in Haryana)
DL16|Faridabad (NCR, situated in Haryana)
DL17|Noida (NCR, situated in Uttar Pradesh)
DL18|Ghaziabad (NCR, situated in Uttar Pradesh)
DN09|Silvassa
GA01|Panaji
GA02|Margao
GA03|Mapusa
GA04|Bicholim
GA05|Ponda
GA06|Vasco da Gama
GA07|Panaji
GA08|Margao
GA09|Quepem
GA10|Canacona
GA11|Pernem
GJ01|Ahmedabad
GJ02|Mehsana
GJ03|Rajkot
GJ04|Bhavnagar
GJ05|Surat
GJ06|Vadodara
GJ07|Kheda
GJ08|Palanpur
GJ09|Sabarkantha
GJ10|Jamnagar
GJ11|Junagadh
GJ12|Kutch
GJ13|Surendranagar
GJ14|Amreli
GJ15|Valsad
GJ16|Bharuch
GJ17|Panchmahal (Godhra)
GJ18|Gandhinagar
GJ19|Bardoli
GJ20|Dahod
GJ21|Navsari
GJ22|Narmada
GJ23|Anand
GJ24|Patan
GJ25|Porbandar
GJ26|Vyara
GJ27|East Ahmedabad
GJ28|West Surat
GJ29|Vadodara rural
GJ30|East Rajkot
HP01|Shimla Urban (tourist buses & taxis)
HP02|Shimla Urban (tourist buses & taxis)
HP03|Shimla Urban
HP04|Dharamsala (auto rickshaws)
HP05|Mandi (auto rickshaws)
HP06|Rampur Bushahr
HP07|Shimla
HP08|Chaupal
HP09|Theog
HP10|Rohru
HP11|Arki
HP12|Nalagarh
HP13|Kandaghat
HP14|Solan
HP15|Parwanoo
HP16|Rajgarh
HP17|Paonta Sahib
HP18|Nahan
HP19|Amb
HP20|Una
HP21|Barsar
HP22|Hamirpur
HP23|Ghumarwin
HP24|Bilaspur
HP25|Kalpa
HP26|Nichar (Bhaba Nagar)
HP27|Pooh
HP28|Sarkaghat
HP29|Jogindernagar
HP30|Karsog
HP31|Sundernagar
HP32|Mandi
HP33|Mandi
HP34|Kullu
HP35|Anni
HP36|Dehra
HP37|Palampur
HP38|Nurpur
HP39|Dharamshala
HP40|Kangra
HP41|Kaza
HP42|Keylong
HP43|Udaipur
HP44|Churah
HP45|Pangi
HP46|Bharmour
HP47|Dalhousie
HP48|Chamba
HP49|Banjar
HP50|Shimla (auto rickshaws)
HP51|Shimla (rural)
HP52|Shimla (rural)
HP53|Baijnath
HP54|Jawali
HP55|Nadaun / Hamirpur
HP56|Jaisinghpur
HP57|Chowari
HP58|Manali
HP62|Dodra Kwar
HP63|Shimla
HP64|Solan
HP65|Mandi
HP66|Kullu
HP67|Hamirpur
HP68|Kangra
HP69|Bilaspur
HP71|Nahan
HP72|Una
HP73|Chamba
HP74|Bhoranj
HP76|Paddhar
HR01|Ambala
HR02|Yamuna Nagar
HR03|Panchkula
HR04|Naraingarh
HR05|Karnal
HR06|Panipat
HR07|Kurukshetra
HR08|Kaithal
HR09|Guhla
HR10|Sonepat
HR11|Gohana
HR12|Rohtak
HR13|Bahadurgarh
HR14|Jhajjar
HR15|Meham
HR16|Bhiwani
HR17|Siwani
HR18|Loharu
HR19|Charkhi Dadri
HR20|Hisar
HR21|Hansi
HR22|Fatehabad
HR23|Tohana
HR24|Sirsa
HR25|Dabwali
HR26|Gurgaon
HR27|Nuh
HR28|Ferozepur Jhirka
HR29|Ballabgarh
HR30|Palwal
HR31|Jind
HR32|Narwana
HR33|Safidon
HR34|Mohindergarh
HR35|Narnaul
HR36|Rewari
HR37|Ambala (commercial vehicles)
HR38|Faridabad (commercial vehicles)
HR39|Hisar (commercial vehicles)
HR40|Assandh
HR41|Pehowa
HR42|Ganaur
HR43|Kosli
HR44|Ellenabad
HR45|Karnal (commercial vehicles)
HR46|Rohtak (commercial vehicles)
HR47|Rewari (commercial vehicles)
HR48|Tosham
JH01|Ranchi
JH02|Hazaribagh / Ramgarh
JH03|Daltonganj
JH04|Dumka
JH05|Jamshedpur
JH06|Chaibasa
JH07|Gumla
JH08|Lohardaga
JH09|Bokaro Steel City
JH10|Dhanbad
JH11|Giridih
JH12|Koderma
JH13|Chatra
JH14|Garhwa
JH15|Deoghar
JH16|Pakur
JH17|Godda
JH18|Sahibganj
JH19|Latehar
JH20|Simdega
JH21|Jamtara
JH22|Saraikela-Kharsawan
JK01|Srinagar
JK02|Jammu
JK03|Anantnag
JK04|Budgam
JK05|Baramulla
JK06|Doda
JK07|Kargil
JK08|Kathua
JK09|Kupwara
JK10|Leh
JK11|Rajouri
JK12|Poonch
JK13|Pulwama
JK14|Udhampur
JK15|Bandipora
JK16|Ganderbal
JK17|Kishtwar
JK18|Kulgam
JK19|Ramban
JK20|Reasi
JK21|Samba
JK22|Shopian
KA01|Bangalore Central (Koramangala)
KA02|Bangalore West (Rajajinagar)
KA03|Bangalore East (Indiranagar)
KA04|Bangalore North (Yeshwanthpur)
KA05|Bangalore South (Jayanagar)
KA06|Tumkur
KA07|Kolar
KA08|Kolar Gold Fields
KA09|Mysore West
KA10|Chamrajnagar
KA11|Mandya
KA12|Madikeri
KA13|Hassan
KA14|Shimoga
KA15|Sagara
KA16|Chitradurga
KA17|Davangere
KA18|Chickmagalur
KA19|Mangalore
KA20|Udupi
KA21|Puttur / Sullia
KA22|Belgaum
KA23|Chikkodi
KA24|Bailhongal
KA25|Dharwad
KA26|Gadag
KA27|Haveri
KA28|Bijapur
KA29|Bagalkot
KA30|Karwar
KA31|Sirsi
KA32|Gulbarga
KA33|Yadgir
KA34|Bellary
KA35|Hospet
KA36|Raichur
KA37|Koppal
KA38|Bidar
KA39|Bhalki
KA40|Chikkaballapur
KA41|Rajarajeshwari Nagar, Bangalore Urban
KA42|Ramanagara
KA43|Devanahalli, Bangalore Rural
KA44|Tiptur
KA45|Hunsur
KA46|Sakleshpur
KA47|Honnavar
KA48|Jamkhandi
KA49|Gokak
KA50|Yelahanka, Bangalore Urban
KA51|Electronics City, Bangalore Urban
KA52|Nelamangala, Bangalore Urban
KA53|K.R. Puram, Bangalore Urban
KA54|Nagamangala
KA55|Mysore East
KA56|Basavakalyan
KA57|Shantinagar, Bangalore Urban
KL01|Thiruvananthapuram City
KL02|Kollam
KL03|Pathanamthitta
KL04|Alappuzha
KL05|Kottayam
KL06|Idukki
KL07|Ernakulam
KL08|Thrissur
KL09|Palakkad
KL10|Malappuram
KL11|Kozhikode
KL12|Wayanad
KL13|Kannur
KL14|Kasargode
KL15|Thiruvananthapuram (KSRTC buses)
KL16|Thiruvananthapuram - Attingal
KL17|Muvattupuzha
KL18|Vadakara
KL19|Thiruvananthapuram - Parassala
KL20|Thiruvananthapuram - Neyyattinkara
KL21|Thiruvananthapuram - Nedumangad
KL22|Kazhakkoottam
KL23|Karunagapally
KL24|Kottarakara
KL25|Punalur
KL26|Adoor
KL27|Thiruvalla
KL28|Mallappally
KL29|Kayamkulam
KL30|Chengannur
KL31|Mavelikara
KL32|Cherthala
KL33|Changanassery
KL34|Kanjirappally
KL35|Pala
KL36|Vaikom
KL37|Vandiperiyar
KL38|Thodupuzha
KL39|Thripunithura
KL40|Perumbavoor
KL41|Aluva
KL42|North Paravur
KL43|Mattancherry
KL44|Kothamangalam
KL45|Irinjalakuda
KL46|Guruvayur
KL47|Kodungallur
KL48|Wadakkancherry
KL49|Alathur
KL50|Mannarghat
KL51|Ottappalam
KL52|Pattambi
KL53|Perinthalmanna
KL54|Ponnani
KL55|Tirur
KL56|Koyilandi
KL57|Koduvally
KL58|Thalassery
KL59|Thaliparamba
KL60|Kanhangad
KL61|Kunnathur
KL62|Ranni
KL63|Angamaly
KL64|Chalakkudy
KL65|Thirurangadi
LD01|Kavaratti
MH01|Mumbai Central (Tardeo)
MH02|Mumbai West (Andheri West)
MH03|Mumbai East (Wadala)
MH04|Thane
MH05|Thane Central Suburbs (Kalyan)
MH06|Raigad (Pen)
MH07|Sindhudurg
MH08|Ratnagiri
MH09|Kolhapur
MH10|Sangli
MH11|Satara
MH12|Pune
MH13|Solapur
MH14|Pune Outskirts (Pimpri-Chinchwad)
MH15|Nashik
MH16|Ahmednagar
MH17|Shrirampur
MH18|Dhule
MH19|Jalgaon
MH20|Aurangabad
MH21|Jalna
MH22|Parbhani
MH23|Beed
MH24|Latur
MH25|Osmanabad
MH26|Nanded
MH27|Amravati
MH28|Buldana
MH29|Yavatmal
MH30|Akola
MH31|Nagpur
MH32|Wardha
MH33|Gadchiroli
MH34|Chandrapur
MH35|Gondia
MH36|Bhandara
MH37|Washim
MH38|Hingoli
MH39|Nandurbar
MH40|Wadi, Nagpur (rural)
MH41|Malegaon
MH42|Baramati
MH43|Navi Mumbai (Vashi)
MH44|Ambejogai
MH45|Akluj
MH46|Panvel
MH47|Borivali, Mumbai
MH48|Vasai, Thane (Virar)
MH49|Nagpur East
MH50|Karad (Satara rural)
ML01|Shillong (government vehicles)
ML02|Shillong (police vehicles)
ML03|Shillong (transport corporation vehicles)
ML04|Jaintia Hills
ML05|East Khasi Hills
ML06|West Khasi Hills
ML07|East Garo Hills
ML08|West Garo Hills
ML09|South Garo Hills
ML10|Ri-Bhoi
MN01|Imphal
MN02|Churachandpur
MN03|Kangpokpi
MN04|Thoubal
MP01|State Governor's vehicle
MP02|Madhya Pradesh Government vehicle
MP03|Madhya Pradesh Police vehicle
MP04|Bhopal
MP05|Hoshangabad
MP06|Morena
MP07|Gwalior
MP08|Guna
MP09|Indore
MP10|Khargone
MP11|Dhar
MP12|Khandwa
MP13|Ujjain
MP14|Mandsaur
MP15|Sagar
MP16|Chhatarpur
MP17|Rewa
MP18|Shahdol
MP19|Satna
MP20|Jabalpur
MP21|Katni
MP22|Seoni
MP28|Chhindwara
MP30|Bhind
MP31|Sheopur
MP32|Datia
MP33|Shivpuri
MP34|Damoh
MP35|Panna
MP36|Tikamgarh
MP37|Sehore
MP38|Raisen
MP39|Rajgarh
MP40|Vidisha
MP41|Dewas
MP42|Shajapur
MP43|Ratlam
MP44|Neemuch
MP45|Jhabua
MP46|Badwani
MP47|Harda
MP48|Betul
MP49|Narsinghpur
MP50|Balaghat
MP51|Mandla
MP52|Dindori
MP53|Sidhi
MP54|Umariya
MP65|Anuppur
MP66|Singrauli
MP67|Ashoknagar
MP68|Burhanpur
MP69|Alirajpur
MZ01|Aizawl
MZ02|Lunglei
MZ03|Saiha
MZ04|Champhai
MZ05|Kolasib
MZ06|Serchhip
MZ07|Lawngtlai
MZ08|Mamit
NL01|Kohima
NL02|Mokokchung
NL03|Tuensang
NL04|Mon
NL05|Wokha
NL06|Zunheboto
NL07|Dimapur
NL08|Phek
OR01|Balasore
OR02|Bhubaneswar
OR03|Bolangir
OR04|Chandikhole
OR05|Cuttack
OR06|Dhenkanal
OR07|Ganjam
OR08|Kalahandi
OR09|Keonjhar
OR10|Koraput
OR11|Mayurbhanj
OR12|Phulabani
OR13|Puri
OR14|Rourkela
OR15|Sambalpur
OR16|Sundergarh
OR17|Baragarh
OR18|Rayagada
OR19|Angul
OR20|Gajapati
OR21|Jagatsinghpur
OR22|Bhadrak
OR23|Jharsuguda
OR24|Nabarangpur
OR25|Nayagarh
OR26|Nuapada
OR27|Boudh
OR28|Debagarh
OR29|Kendrapara
OR30|Malkangiri
OR31|Sonepur
PB01|Punjab (taxis)
PB02|Amritsar
PB03|Bathinda
PB04|Faridkot
PB05|Ferozepur
PB06|Gurdaspur
PB07|Hoshiarpur
PB08|Jalandhar
PB09|Kapurthala
PB10|Ludhiana
PB11|Patiala
PB12|Ropar
PB13|Sangrur
PB14|Ajnala
PB15|Abohar
PB16|Anandpur Sahib
PB17|Baba Bakala
PB18|Batala
PB19|Barnala
PB20|Balachaur
PB21|Dasua
PB22|Fazilka
PB23|Fatehgarh Sahib
PB24|Garhshankar
PB25|Jagraon
PB26|Khanna
PB27|Kharar
PB28|Malerkotla
PB29|Moga
PB30|Muktsar
PB31|Mansa
PB32|Nawanshahr
PB33|Nakodar
PB34|Nabha
PB35|Pathankot
PB36|Phagwara
PB37|Phillaur
PB38|Patti
PB39|Rajpura
PB40|Rampura Phul
PB41|Sultanpur Lodhi
PB42|Samana
PB43|Samrala
PB44|Sunam
PB45|Talwandi Sabo
PB46|Tarn Taran
PB47|Zira
PB48|Amloh
PB49|Khamano
PB50|Budhlada
PB51|Jhunir / Sardulgarh
PB52|Bassi Pathana
PB53|Malout
PB54|Mukerian
PB55|Payal
PB56|Raikot
PB57|Bhulath
PB58|Dera Baba Nanak
PB59|Dhuri
PB60|Gidderbaha
PB61|Jalalabad
PB62|Jaitu
PB63|Khadoor Sahib
PB64|Moonak
PB65|Mohali
PB66|Nihal Singh Wala
PB67|Shahkot
PB68|Dhar Kalan
PB69|Bagha Purana
PB70|Dera Bassi
PB71|Chamkaur Sahib
PB72|Patran
PB73|Tappa Mandi
PB74|Nangal
PB75|Lehra
PY01|Puducherry
PY02|Karaikal
PY03|Mahe
PY04|Yanam
RJ01|Ajmer
RJ02|Alwar
RJ03|Banswara
RJ04|Barmer
RJ05|Bharatpur
RJ06|Bhilwara
RJ07|Bikaner
RJ08|Bundi
RJ09|Chittaurgarh
RJ10|Churu
RJ11|Dholpur
RJ12|Dungarpur
RJ13|Sri Ganganagar
RJ14|Jaipur
RJ15|Jaisalmer
RJ16|Jalore
RJ17|Jhalawar
RJ18|Jhunjhunu
RJ19|Jodhpur
RJ20|Kota
RJ21|Nagaur
RJ22|Pali
RJ23|Sikar
RJ24|Sirohi
RJ25|Sawai Madhopur
RJ26|Tonk
RJ27|Udaipur
RJ28|Baran
RJ29|Dausa
RJ30|Rajsamand
RJ31|Hanumangarh
RJ32|Kotputli
RJ33|Ramganj Mandi
RJ34|Karauli
RJ35|Pratapgarh
RJ36|Beawar
RJ37|Didwana
SK01|Gangtok, East Sikkim
SK02|Gyalshing, West Sikkim
SK03|Mangan, North Sikkim
SK04|Jorethang, South Sikkim
TN01|Chennai Central (Ayanavaram)
TN02|Chennai West (Anna Nagar)
TN03|Chennai North East (Tondiarpet)
TN04|Chennai East (Basin Bridge)
TN05|Chennai North (Vyasarpadi)
TN06|Chennai South East (Mandaveli)
TN07|Chennai South (Thiruvanmiyur)
TN09|Chennai West (K.K. Nagar)
TN10|Chennai South West (Valasaravakkam)
TN11|Tambaram
TN16|Tindivanam
TN18|Sengundram
TN19|Chengalpattu
TN20|Tiruvallur
TN21|Kanchipuram
TN22|Meenambakkam
TN23|Vellore
TN24|Krishnagiri
TN25|Thiruvannamalai
TN28|Namakkal
TN29|Dharmapuri
TN30|Salem West
TN31|Cuddalore
TN32|Villupuram
TN33|Erode
TN34|Thiruchengode
TN36|Gobichettipalayam
TN37|Coimbatore South
TN38|Coimbatore North
TN39|Tirupur North
TN40|Mettupalayam
TN41|Pollachi
TN42|Tirupur South
TN43|Udhagamandalam
TN45|Tiruchirapalli
TN46|Perambalur
TN47|Karur
TN48|Srirangam
TN49|Thanjavur
TN50|Tiruvarur
TN51|Nagapattinam
TN52|Sankagiri
TN54|Salem East
TN55|Pudukottai
TN56|Perundurai
TN57|Dindigul
TN58|Madurai South
TN59|Madurai North
TN60|Theni
TN61|Ariyalur
TN63|Sivagangai
TN64|Madurai Central
TN65|Ramanathapuram
TN66|Coimbatore Central
TN67|Virudhunagar
TN68|Kumbakonam
TN69|Tuticorin
TN70|Hosur
TN72|Tirunelveli
TN73|Ranipet
TN74|Nagercoil
TN75|Marthandam
TN76|Tenkasi
TN77|Attur
TN78|Dharapuram
TR01|Agartala
TR02|Kailashahar
TR03|Udaipur
UA07|Dehradun
UK01|Almora
UK02|Bageshwar
UK03|Champawat
UK04|Nainital
UK05|Pithoragarh
UK06|Udham Singh Nagar
UK07|Dehradun
UK08|Haridwar
UK09|Tehri
UK10|Uttarkashi
UK11|Chamoli
UK12|Pauri
UK13|Rudraprayag
UP11|Saharanpur
UP12|Muzaffarnagar
UP13|Bulandshahr
UP14|Ghaziabad
UP15|Meerut
UP16|Noida / Gautam Buddha Nagar
UP17|Baghpat
UP18|Prabudh Nagar
UP20|Bijnor
UP21|Moradabad
UP22|Rampur
UP23|Jyotiba Phule Nagar
UP24|Budaun
UP25|Bareilly
UP26|Pilibhit
UP27|Shahjahanpur
UP28|Ayodhya
UP30|Hardoi
UP31|Kheri
UP32|Lucknow
UP33|Raebareli
UP34|Sitapur
UP35|Unnao
UP36|Lucknow
UP37|Hapur
UP40|Bahraich
UP41|Barabanki
UP42|Faizabad
UP43|Gonda
UP44|Sultanpur
UP45|Ambedkar Nagar
UP46|Shrawasti
UP47|Balrampur
UP50|Azamgarh
UP51|Basti
UP52|Deoria
UP53|Gorakhpur
UP54|Mau
UP55|Siddharth Nagar
UP56|Mahrajganj
UP57|Padrauna
UP58|Sant Kabir Nagar
UP60|Ballia
UP61|Ghazipur
UP62|Jaunpur
UP63|Mirzapur
UP64|Sonbhadra
UP65|Varanasi
UP66|Bhadohi
UP67|Chandauli
UP70|Prayagraj (Allahabad)
UP71|Fatehpur
UP72|Pratapgarh
UP73|Kaushambi
UP74|Kannauj
UP75|Etawah
UP76|Farrukhabad
UP77|Kanpur Dehat
UP78|Kanpur Urban
UP79|Auraiya
UP80|Agra
UP81|Aligarh
UP82|Etah
UP83|Firozabad
UP84|Mainpuri
UP85|Mathura
UP86|Mahamaya Nagar
UP87|Kanshiram Nagar
UP90|Banda
UP91|Hamirpur
UP92|Jalaun
UP93|Jhansi
UP94|Lalitpur
UP95|Mahoba
UP96|Chitrakoot Dham (Karwi)
WB01|Kolkata (two-wheelers)
WB02|Kolkata (private four-wheelers)
WB03|Kolkata (goods carriages)
WB04|Kolkata (passenger vehicles)
WB05|Kolkata (tractors, trailers, cranes)
WB06|Kolkata (private four-wheelers)
WB07|Kolkata (two-wheelers)
WB08|Kolkata
WB09|Kolkata
WB10|Kolkata
WB11|Howrah (transport vehicles)
WB12|Howrah
WB13|Howrah
WB14|Howrah
WB15|Hooghly (transport vehicles)
WB16|Hooghly
WB18|Hooghly
WB19|South 24 Parganas, Alipore (transport vehicles)
WB20|South 24 Parganas, Alipore
WB22|South 24 Parganas, Alipore
WB23|Barrackpore (transport vehicles)
WB24|Barrackpore
WB25|Barasat (transport vehicles)
WB26|Barasat
WB29|Tamluk (transport vehicles)
WB30|Tamluk
WB31|Contai (transport vehicles)
WB32|Contai
WB33|Midnapur (transport vehicles)
WB34|Midnapur
WB37|Asansol (transport vehicles)
WB38|Asansol
WB39|Durgapur (transport vehicles)
WB40|Durgapur
WB41|Burdwan (transport vehicles)
WB42|Burdwan
WB44|Asansol
WB51|Nadia (transport vehicles)
WB52|Nadia
WB53|Birbhum (transport vehicles)
WB54|Birbhum
WB55|Purulia (transport vehicles)
WB56|Purulia
WB57|Murshidabad (transport vehicles)
WB58|Murshidabad
WB59|Raiganj (transport vehicles)
WB60|Raiganj
WB61|Balurghat (transport vehicles)
WB62|Balurghat
WB63|Cooch Behar (transport vehicles)
WB64|Cooch Behar
WB65|Malda (transport vehicles)
WB66|Malda
WB67|Bankura (transport vehicles)
WB68|Bankura
WB69|Alipurduar (transport vehicles)
WB70|Alipurduar
WB71|Jalpaiguri (transport vehicles)
WB72|Jalpaiguri
WB73|Siliguri (transport vehicles)
WB74|Siliguri
WB76|Darjeeling (transport vehicles)
WB77|Darjeeling
WB78|Kalimpong (transport vehicles)
WB79|Kalimpong
"""

def _build() -> dict[str, str]:
    table: dict[str, str] = {}
    for line in _DISTRICT_TABLE.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        code, _, place = line.partition("|")
        table[code.strip().upper()] = place.strip()
    return table


DISTRICTS: dict[str, str] = _build()


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

#: A plate begins with two letters and then the district digits. Anything
#: after that is the series and the vehicle number, which carry no
#: geography, so the pattern deliberately stops caring at that point.
_PREFIX = re.compile(r"^\s*([A-Z]{2})\s*([0-9]{1,2})?")

#: Letters an OCR engine returns where a digit belongs. The district
#: position of an Indian plate is digits and nothing else, so a letter
#: appearing there is a misread, not an alternative spelling - which is
#: what makes it safe to correct. "HPO1" becomes "HP01" (Shimla); it could
#: not have been a real code either way.
#:
#: Applied ONLY to the one or two characters in the district position.
#: Nowhere else: the series letters after it are genuinely letters, and
#: "correcting" those would turn a good read into a wrong plate.
_DIGIT_CONFUSIONS = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


def _as_district_digits(text: str) -> Optional[str]:
    """Read the district code, repairing letter-for-digit misreads.

    Returns None when the characters cannot be a district code even after
    correction, so a bad read fails rather than being forced into a number.
    """
    out = []
    for ch in text[:2]:
        if ch.isdigit():
            out.append(ch)
        elif ch in _DIGIT_CONFUSIONS:
            out.append(_DIGIT_CONFUSIONS[ch])
        else:
            break
    return "".join(out) or None

#: The Bharat series is written 22BH1234A: two year digits, then BH.
_BHARAT = re.compile(r"^\s*[0-9]{2}\s*BH", re.IGNORECASE)


@dataclass(frozen=True)
class RtoOrigin:
    """Where a plate was registered, and how much of that is actually known."""

    plate: str
    state_code: Optional[str]
    state: Optional[str]
    district_code: Optional[str]
    district: Optional[str]

    @property
    def known(self) -> bool:
        """True when at least the state was identified."""
        return self.state is not None

    @property
    def district_known(self) -> bool:
        return self.district is not None

    def summary(self) -> str:
        """One line for an operator, honest about what is missing.

        Never returns a half-sentence with a blank in it: if the district is
        not in the table it says so, because "Punjab, " reads like a bug and
        "Punjab (district not in table)" reads like an answer.
        """
        if not self.state:
            return "Origin unknown"
        if self.district:
            return f"{self.district}, {self.state}"
        if self.state_code in NATIONWIDE:
            return self.state
        if self.district_code:
            return f"{self.state} (district {self.district_code} not in table)"
        return f"{self.state} (no district code read)"


def decode(plate: Optional[str]) -> RtoOrigin:
    """Decode a plate string into its registering state and district.

    Tolerant of the spacing and case a reader may produce ("PB 39 N 0203",
    "pb39n0203"), because the OCR output is not under this function's
    control. Everything else is strict: an unrecognised state prefix
    returns nothing rather than a nearest guess, and a district code that
    is not in the table returns None for the district.
    """
    if not plate:
        return RtoOrigin("", None, None, None, None)

    raw = str(plate).strip()
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    if not cleaned:
        return RtoOrigin(raw, None, None, None, None)

    # The Bharat series puts the year first, so the usual "first two
    # letters are the state" rule would read the digits and fail.
    if _BHARAT.match(cleaned):
        return RtoOrigin(raw, "BH", STATES["BH"], None, None)

    match = _PREFIX.match(cleaned)
    if not match:
        return RtoOrigin(raw, None, None, None, None)

    state_code = match.group(1)
    state = STATES.get(state_code)
    if state is None:
        # Not a recognised prefix. Almost always a misread rather than a
        # real plate, and inventing a state for it would be worse than
        # saying nothing.
        return RtoOrigin(raw, None, None, None, None)

    digits = match.group(2)
    if not digits:
        # The regex wanted digits and did not find them. Before giving up,
        # try reading the same position as misread digits - an OCR engine
        # returning "HPO1" for "HP01" is the single most common way a
        # perfectly good plate loses its district.
        digits = _as_district_digits(cleaned[2:])
    if not digits:
        return RtoOrigin(raw, state_code, state, None, None)

    # Delhi is written DL1 through DL18 without a leading zero, every other
    # state pads to two digits. Try the plate's own form first, then the
    # padded form, so both spellings find the same row.
    for candidate in (f"{state_code}{digits}", f"{state_code}{int(digits):02d}"):
        district = DISTRICTS.get(candidate)
        if district is not None:
            return RtoOrigin(raw, state_code, state, candidate, district)

    return RtoOrigin(raw, state_code, state, f"{state_code}{digits}", None)


def state_of(plate: Optional[str]) -> Optional[str]:
    """Just the state, for a caller that wants one string."""
    return decode(plate).state


def district_of(plate: Optional[str]) -> Optional[str]:
    """Just the district, or None when it is not in the table."""
    return decode(plate).district


def coverage() -> dict[str, int]:
    """How much data this module actually holds.

    Exposed so the interface can state the size of the table rather than
    implying it is exhaustive.
    """
    return {
        "states": len(set(STATES.values())),
        "state_codes": len(STATES),
        "district_codes": len(DISTRICTS),
    }
