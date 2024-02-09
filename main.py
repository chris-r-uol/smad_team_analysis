import streamlit as st
import scraper
import graph 
from collections import Counter
st.set_page_config(layout='wide')
st.title('Web Scraper and CRM Analysis')

if 'input_model' in st.session_state == False:
    st.session_state['input_model'] = None
if 'colour_gradient' in st.session_state == False:
    st.session_state['colour_gradient'] = 'Portland'

which_model = st.selectbox('Select Model', options = ['distilroberta-base'])

st.session_state['input_model'] = scraper.load_model('distilroberta-base')

st.header('Single Profile Analysis')
st.info('You can type the link of a University Profile in the box provided and it will automatically generate a list of interests.')
url = st.text_input('Input URL here:', value = 'https://environment.leeds.ac.uk/transport/staff/975/professor-simon-shepherd')
try:
    profile = scraper.profile_from_link(url)
    st.write(profile)
except:
    pass

urls = ['https://environment.leeds.ac.uk/transport/staff/975/professor-simon-shepherd',
        'https://environment.leeds.ac.uk/transport/staff/8835/dr-chris-rushton',
        'https://environment.leeds.ac.uk/transport/staff/980/dr-james-tate',
        'https://environment.leeds.ac.uk/transport/staff/8838/dr-joey-talbot',
        'https://environment.leeds.ac.uk/transport/staff/2506/dr-li-ke-jiang',
        'https://environment.leeds.ac.uk/transport/staff/941/dr-yue-huang',
        'https://environment.leeds.ac.uk/transport/staff/963/dr-dave-milne',
        'https://environment.leeds.ac.uk/transport/staff/9887/dr-angelica-salas-jones',
        'https://environment.leeds.ac.uk/transport/staff/917/dr-chandra-balijepalli',
        'https://environment.leeds.ac.uk/transport/staff/923/professor-haibo-chen',
        'https://environment.leeds.ac.uk/transport-social-political-sciences/staff/9633/eugeni-vidal-tortosa',
        'https://environment.leeds.ac.uk/transport/staff/915/professor-jillian-anable',
        'https://environment.leeds.ac.uk/transport/staff/2728/dr-jo-ann-pattinson',
        'https://environment.leeds.ac.uk/transport/staff/1831/dr-judith-wang',
        'https://environment.leeds.ac.uk/transport/staff/974/dr-karl-ropkins',
        'https://environment.leeds.ac.uk/transport/staff/982/dr-paul-timms',
        'https://environment.leeds.ac.uk/transport/staff/951/professor-ronghui-liu',
        'https://environment.leeds.ac.uk/transport/staff/932/professor-susan-grant-muller',
        'https://environment.leeds.ac.uk/transport/staff/9273/dr-ye-liu',
        'https://environment.leeds.ac.uk/transport/staff/918/dr-yvonne-barnard',
        'https://environment.leeds.ac.uk/transport/staff/12594/dr-afzal-ahmed'
        ]

st.header('Graphs')
which_colour_grade = st.selectbox('Select Colour Option', options = sorted(['Viridis', 'Cividis', 'Plasma', 'Haline', 'ArmyRose', 'Solar', 'Portland', 'IceFire', 'SunsetDark', 'Thermal', 'Turbid', 'Spectral', 'Twilight', 'Magma', 'Tropic']))
st.session_state['colour_gradient'] = which_colour_grade

profiles = [scraper.profile_from_link(u) for u in urls]
profile_graph = graph.make_graph_from_profiles(profiles)

plot_profile_graph = graph.plot_graph(profile_graph)


#for i in interests:
#    st.write(i)
#st.write(interests)

st.subheader('Graph With People as Nodes')
st.info('The profiles for a range of SMaD members have been automatically inputted and a network of interests has been generated.')
st.plotly_chart(plot_profile_graph, use_container_width=True)

st.subheader('Graph With Interests as Nodes')
interests = list(graph.rearrange_dicts(profiles))
interests_graph = graph.make_graph_from_profiles(interests)
plot_interests_graph = graph.plot_graph(interests_graph)
st.plotly_chart(plot_interests_graph, use_container_width=True)




