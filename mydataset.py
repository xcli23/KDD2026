from torch.utils.data import Dataset
import pandas as pd


class T2IDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.lines = [line.strip() for line in file]  # 去掉换行符

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return {"text": self.lines[idx], "index": idx}
    

# class Dataset(Dataset):
#     def __init__(self, file_path):
#         self.dataset = pd.read_csv(file_path, sep='\t')
#         prompts = self.dataset['Prompt'].tolist()
        
#         self.lines = [line.strip() for line in prompts]  # 去掉换行符

#     def __len__(self):
#         return len(self.lines)

#     def __getitem__(self, idx):
#         return {"text": self.lines[idx], "index": idx}
    
class Dataset(Dataset):
    def __init__(self, csv_path):
        self.dataset = pd.read_csv(csv_path)
        # 检查列名是否存在，优先用 article/highlights
        if 'article' in self.dataset.columns and 'highlights' in self.dataset.columns:
            self.texts = self.dataset['article'].tolist()
            self.references = self.dataset['highlights'].tolist()
        elif 'Prompt' in self.dataset.columns:
            self.texts = self.dataset['Prompt'].tolist()
            self.references = None  # 无 reference 时设为 None
        else:
            raise ValueError("CSV 必须包含 'article/highlights' 或 'Prompt' 列！")
        self.lines = [line.strip() for line in self.texts]  
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.references is not None:
            item["reference"] = self.references[idx]
        return item
    

subsampled_data = {
    'promptist': (
    [
     'A rabbit is wearing a space suit.', 
     'Several railroad tracks with one train passing by.',
     'The roof is wet from the rain.',
     'Cats dancing in a space club.'
    ],
    [
     'A rabbit is wearing a space suit, digital Art, Greg rutkowski, Trending cinematographic artstation.',
     'Several railroad tracks with one train passing by, hyperdetailed, artstation, cgsociety, 8k.',
     'The roof is wet from the rain, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.',
     'Cats dancing in a space club, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, fantasy.'
    ]
    ),
    'beautiful': (
    [
     'Astronaut rides horse.', 
     'A majestic sailing ship.',
     'Sunshine on iced mountain.',
     'Panda mad scientist mixing sparkling chemicals.'
    ],
    [
     'Astronaut riding a horse, fantasy, intricate, elegant, highly detailed, artstation, concept art, smooth, sharp focus, illustration.',
     'A massive sailing ship, by Greg Rutkowski, highly detailed, stunning beautiful photography, unreal engine, 8K.',
     'Photo of sun rays coming from melting iced mountain, by greg rutkowski, 4 k, trending on artstation.',
     'Panda as a mad scientist, lab coat, mixing glowing and disinertchemicals, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.'
    ]
    ),
    'cnn_dailymail': (
    [
    #  "Nasa has warned of an impending asteroid pass - and says it will be the closest until 2027. The asteroid, designated 2004 BL86, will safely pass about three times the distance of Earth to the moon on January 26. It will be the closest by any known space rock this large until asteroid 1999 AN10 flies past Earth in 2027. See the Asteroid's route below . At the time of its closest approach on January 26, the asteroid will be approximately 745,000 miles (1.2 million kilometers) from Earth. Due to its orbit around the sun, the asteroid is currently only visible by astronomers with large telescopes who are located in the southern hemisphere. But by Jan. 26, the space rock's changing position will make it visible to those in the northern hemisphere. From its reflected brightness, astronomers estimate that the asteroid is about a third of a mile (0.5 kilometers) in size. At the time of its closest approach on January 26, the asteroid will be approximately 745,000 miles (1.2 million kilometers) from Earth. 'Monday, January 26 will be the closest asteroid 2004 BL86 will get to Earth for at least the next 200 years,' said Don Yeomans, who is retiring as manager of NASA's Near Earth Object Program Office at the Jet Propulsion Laboratory in Pasadena, California, after 16 years in the position. 'And while it poses no threat to Earth for the foreseeable future, it's a relatively close approach by a relatively large asteroid, so it provides us a unique opportunity to observe and learn more.' One way NASA scientists plan to learn more about 2004 BL86 is to observe it with microwaves. NASA's Deep Space Network antenna at Goldstone, California, and the Arecibo Observatory in Puerto Rico will attempt to acquire science data and radar-generated images of the asteroid during the days surrounding its closest approach to Earth. 'When we get our radar data back the day after the flyby, we will have the first detailed images,' said radar astronomer Lance Benner of JPL, the principal investigator for the Goldstone radar observations of the asteroid. 'At present, we know almost nothing about the asteroid, so there are bound to be surprises.' Don't Panic! Nasa says 'While it poses no threat to Earth for the foreseeable future, it's a relatively close approach by a relatively large asteroid, so it provides us a unique opportunity to observe and learn more.' Asteroid 2004 BL86 was initially discovered on Jan. 30, 2004 by a telescope of the Lincoln Near-Earth Asteroid Research (LINEAR) survey in White Sands, New Mexico. The asteroid is expected to be observable to amateur astronomers with small telescopes and strong binoculars. 'I may grab my favorite binoculars and give it a shot myself,' said Yeomans. 'Asteroids are something special. Not only did asteroids provide Earth with the building blocks of life and much of its water, but in the future, they will become valuable resources for mineral ores and other vital natural resources. 'They will also become the fueling stops for humanity as we continue to explore our solar system. There is something about asteroids that makes me want to look up.", 
    #  "By . David Kent . Andy Carroll has taken an understandably glum-looking selfie in hospital. The striker took to Instagram to post a snap of himself just as he prepared to go in for ankle surgery. And you can understand the look on his face, after several injury-ravaged seasons with Liverpool and West Ham, Carroll was finally hoping for a clean bill of health as he finally looked to fulfill the promise he showed following his breakthrough at Newcastle.VIDEO Scroll down to see behind the scenes of West Ham's new kit unveiling . Glum: Andy Carroll's pre-operation selfie from hospital . Blow: Andy Carroll is reportedly out for four months after tearing ligaments in his ankle . SEPTEMBER 2012Misses a month after straining a hamstring against FulhamDECEMBER 2012Twists his knee against Man Utd and is out for two monthsMAY 2013Falls awkwardly and damages his heel against Reading and is out for seven months . But, once again, it wasn't to be, as an ankle injury on a tour of New Zealand ruled him out for up to four months. It is then expected that he will not . return to action until December at the earliest, missing the crucial . start of West Ham's campaign. The . England international was unable to play last season until January after . injuring his left heel and made only 16 appearances. That . injury had completely healed and he was gearing up for the start of the . new campaign by heading out to New Zealand for the club's pre-season . tour when this fresh injury struck in training. It is understood to be . completely unrelated to the one he sustained 14 months ago. In . the last game of the 2012/13 season he ruptured the lateral plantar . fascia - the tendons that run through the arch of the foot, connecting . the heel to the toes - in his right foot leaping for a ball against . Reading. When he was thought to be on the mend, the medial one went and he was out for almost six months of the last campaign. Having . been on loan at the club from Liverpool for one season, West Ham . pressed ahead to make the transfer permanent last summer in what Sam . Allardyce described as a 'calculated risk.' Missing: Carroll had to withdrawn from West Ham's pre-season tour in New Zealand . Out: Carroll was originally expected to recover from the injury before the start of next season . The . West Ham manager was initially told that Carroll, who has won nine caps . for England, would only be out until September but he ended up missing . the entire first half of the season. This . was despite owner David Sullivan seeking out a top specialist, Lieven . Maesschalck, and he spent time with the physiotherapist in Belgium to . help his recovery. It will be hugely frustrating for Allardyce that he will have to start a second successive season without his star striker. Carroll . has scored just two goals since he became their record signing and, on . around £80,000 per week, their highest-paid player. He has so far cost . them £1million per Premier League game. Fans . will be hoping that this is the last of his injury problems, with his . contract set to run for another five years until 2019.",
    #  "Los Angeles (CNN) -- Los Angeles has long been a destination for artistic dreamers from Europe: Zsa Zsa Gabor moved to Hollywood from Hungary in the 1940s to act. Warsaw-born Roman Polanski moved to Southern California in the 1960s to direct. Not to mention one ambitious actor named Arnold Schwarzenegger, who arguably has done more to boost California's image as a place receptive to Europeans than any tourism initiative the state might have dreamed up the past 30 years. But for accented aspiring pop stars from the EU and beyond, L.A. hasn't generally been considered the place to launch an international music career. That honor fell to cities such as London and New York. Until now. These days Manhattan is getting the flyover treatment as singers from all over Europe and farther east set their sights on the U.S. market via Hollywood as the new must-conquer gateway to American ears and eyes. Artists such as Estonia's Kerli, Italy's Marco Bosco, t.A.T.u.'s Lena Katina from Russia, Slovakia's TWiiNS and Austria's Fawni are suddenly swarming L.A. with dreams of making it big. Their presence is being felt at small clubs such as the Troubadour (Katina played a solo show at the venue last year) to red carpets (Fawni is now well known to Hollywood event photographers) to purchased billboards on Sunset Boulevard (Bosco recently bought expensive outdoor media to promote himself along the busy, high visibility corridor). ""I love being here...Los Angeles is my second home now,"" says Katina, who is working on her first solo record and now splits her time between L.A. and Moscow. Katina and other singers from Russia and Europe's timing couldn't be better: America has finally started to embrace the increasing globalization of pop music on a scale beyond the occasional super group (see ABBA) or German one-hit wonder (see Nena's ""99 Luftballoons"") thanks largely to Websites such as YouTube, which has leveled the playing field and cut out past gatekeepers such as MTV. Swedish singer Robyn topped many critical lists in 2010, with Denmark's Medina set to make similar inroads in the United States this year with early adopters in the pop and dance music arenas. But perhaps the most interesting singer ready to make the crossover in 2011 is Estonia's Kerli. ""When I first got here, someone told me 'there are no friends in the music business' and I was so hurt,"" the former winner of a Baltic version of ""American Idol"" said over coffee at a West Hollywood restaurant last month. ""But Los Angeles is an amazing place to live once you find the people that inspire you, and I've found that circle of friends here,"" the singer said. ""We make art together, and we constantly feed off each other."" The blonde beauty who looks like a glammed-out Goth version of Lady Gaga (though she and her fans loathe the comparison) and sounds like a hybrid of Bjork, Brandy and Avril Lavigne moved to L.A. around four years ago and has been slowly winning over American fans ever since. Her debut for Island Records, 2008's ""Love Is Dead,"" did fairly well for a new artist, considering Kerli is pushing a sound she herself calls ""Bubblegoth."" According to Nielsen SoundScan, around 65,000 copies were sold. However, both the singer and her label are thinking bigger this year after buzz surrounding her just-released ""Army Of Love"" began heating up the internet. It's too soon to tell if mainstream pop radio stations will embrace Kerli in 2011 (her follow-up full length record is expected to see a release by summer), but there are encouraging signs. AOL's popular Popeater blog featured the singer late last year in a campaign worthy of a former ""American Idol"" star; rolling out her video for the released-in-December ""Army Of Love"" with video diaries building up to a December 22 premiere. ""It's like Euro trash meets angels singing in a choir,"" Kerli said of ""Army Of Love,"" which continues to draw interest online because of the video, which has a curious mix of swirling melodies set against striking visuals (the clip was shot in Estonia). Adventurous college radio listeners have long been boosters of acts from the Baltic states and other European countries, but mainstream pop fans rarely hear singers such as Kerli on the U.S. pop charts. And while European artists who have ""made it"" overseas have been buying second homes in the Hollywood Hills for decades, more interesting are the new pop singers living nearby, such as Slovakia's TWiiNS, who are hoping against odds to make a name for themselves in America after a modicum of success elsewhere. The duo, who are identical twins, moved to L.A. last year. They are currently working on their first record for L.A.-based indie label B Records with known U.S. producers including Bryan Todd, who has worked with names such as Jordin Sparks. ""We love Los Angeles because of the weather, nice people and shopping, but the main reason why we moved is our work,"" Veronika Nízlové said via email last month. Her twin sister Daniela added the transition has not been easy. ""It's really hard to come from Eastern Europe and try to achieve success in America. We are not native speakers, we are not Americans...it's a little disadvantage to us, but our big advantage is that we are twins. "" TWiiNS, which scored a minor European hit last year with a remake of Sabrina Salerno's 1980s hit ""Boys,"" seem already savvy to the city's sometimes cruel undercurrent. In their forthcoming single ""Welcome to Hollywood,"" the pair warn other aspiring singers that not everything is sunshine and smiles in the City of Angels. ""The song is not about the perfect Hollywood,"" Veronika said, ""It's about people with their 'friendly faces' which is far from being true. You should have open eyes and be careful whom you trust. Hollywood and all that goes along with it really has two sides to it."" Sage advice from Los Angeles' latest émigrés, who sing on their soon-to-be released single: ""Welcome to Hollywood/Boy you better give it up before it gets you down/Welcome to Hollywood/Just got to get a grip of how to get around.",
    ],
    [
     "2004 BL86 will pass about three times the distance of Earth to the moon .Estimate that the asteroid is about a third of a mile (0.5 kilometers) in size .Nasa says it poses no threat to Earth 'for the foreseeable future",
     "Carroll takes to Instagram to post selfie ahead of ankle surgery . West Ham star expected to be out for up to four months. The forward has had an injury-ravaged spell since moving from Liverpool .",
     "Pop stars from all over Europe are setting their sights on the U.S. market . Estonia's Kerli, Italy's Marco Bosco and Austria's Fawni want to make it big in L.A. Los Angeles has long been a destination for European artists seeking fame .",

    ]
    )
}
# subsampled_data = (
#     ['Astronaut rides horse.', 
#      'A majestic sailing ship.',
#      'Sunshine on iced mountain.',
#      'Panda mad scientist mixing sparkling chemicals.'],
#     ['Astronaut riding a horse, fantasy, intricate, elegant, highly detailed, artstation, concept art, smooth, sharp focus, illustration.',
#      'A massive sailing ship, by Greg Rutkowski, highly detailed, stunning beautiful photography, unreal engine, 8K.',
#      'Photo of sun rays coming from melting iced mountain, by greg rutkowski, 4 k, trending on artstation.',
#      'Panda as a mad scientist, lab coat, mixing glowing and disinertchemicals, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.']
# )