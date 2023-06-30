import pandas as pd
import matplotlib.pyplot as plt

def bar_plot(dict1,dict2, name1, name2):
    df1 = pd.DataFrame(list(dict1.items()), columns=['Label', name1])  # set column name as name1
    df2 = pd.DataFrame(list(dict2.items()), columns=['Label', name2])  # set column name as name2

    df = pd.concat([df1, df2], axis=1)

    df[[name1, name2]].plot(kind='bar', logy=True)  # use new column names for plotting

    plt.xticks(ticks=range(df.shape[0]), labels=list(zip(df1.Label, df2.Label)), rotation=45)

    plt.gcf().subplots_adjust(bottom=0.25)  # Adjust bottom margin

    plt.show()

def count_prompts_with_tag(prompts, tag):
    return sum([1 if tag.lower() in p.lower() else 0 for p in prompts])

def analyse_gender(prompts):
    FEMALE_GENDER_TAGS = [" woman", " women", " girl", " girls", " female", " females"]
    MALE_GENDER_TAGS = [" man", " men", " boy", " boys", " male", " males"]


    female_tag_counts = {tag: count_prompts_with_tag(prompts, tag) for tag in FEMALE_GENDER_TAGS}
    print("female:", female_tag_counts)
    male_tag_counts = {tag: count_prompts_with_tag(prompts, tag) for tag in MALE_GENDER_TAGS}
    print("male:", male_tag_counts)

    bar_plot(female_tag_counts, male_tag_counts, "female", "male")
    total_female = sum(female_tag_counts.values())
    total_male = sum(male_tag_counts.values())
    print("total female:", total_female)
    print("total male:", total_male)

def famous_people(prompts):
    df = pd.read_csv('data/pantheon.tsv', sep='\t')
    # filtering,keeping only US, UK
    desired_values = ['United Kingdom', 'UNITED STATES']
    df_filtered = df[df['countryName'].isin(desired_values)]
    df_filtered_no_style = df_filtered[df_filtered['industry'] != 'FINE ARTS']
    df_filtered_no_style = df_filtered_no_style[df_filtered_no_style['occupation'] != 'FILM DIRECTOR']

    #df_only_style = df[df['industry'] == 'FINE ARTS']
    df_only_style = df[df['occupation'] == 'FILM DIRECTOR']


    famous_name_list = df_filtered_no_style['name'].to_list()
    famous_style_list = df_only_style['name'].to_list()

    '''''
    famous_counts = {tag: count_prompts_with_tag(prompts, tag) for tag in famous_name_list}
    sorted_famous_caounts = {k: v for k, v in sorted(famous_counts.items(), key=lambda item: item[1], reverse=True) if v != 0}
    print("poeple:", sorted_famous_caounts)
    '''
    famous_styles_counts = {tag: count_prompts_with_tag(prompts, tag) for tag in famous_style_list}
    sorted_famous_counts = {k: v for k, v in sorted(famous_styles_counts.items(), key=lambda item: item[1], reverse=True) if v != 0}
    print("poeple:", sorted_famous_counts)

def movie_charecters(prompts):
    df = pd.read_csv('data/mbti.csv')
    #print(df['role'])
    char_list = df['role'].to_list()
    char_counts = {tag: count_prompts_with_tag(prompts, tag) for tag in char_list}
    sorted_char_counts = {k: v for k, v in sorted(char_counts.items(), key=lambda item: item[1], reverse=True) if v != 0}
    print("chars:", sorted_char_counts)


def analyse_tags(prompts):
    analyse_gender(prompts)
    famous_people(prompts)
    movie_charecters(prompts)
