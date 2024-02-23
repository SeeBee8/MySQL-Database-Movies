import streamlit as st

st.set_page_config()

st.title("Ranking Movie Reviews")

st.write('Taking a look at movie reviews, this model takes a review and tells you how the movie will rate.  \n Find out if the movie rates highly or not.')

st.subheader("How will your movie rate?")
st.markdown(
    '''**Select a page from the sidebar**''')



import streamlit.components.v1 as components 
# Source: https://meta.stackoverflow.com/questions/392785/need-to-add-linkedin-and-github-badges-in-profile-page-of-stack-overflow
# Source: https://docs.streamlit.io/library/api-reference/layout/st.container
st.divider()
st.markdown("***Brought to you By: Christina Brockway***")
st.markdown("**Contact Me!**")
links_html = """<ul>
<li><a href="csbrockway602@gmail.com">Email</a></li>
    <li><a href="https://www.linkedin.com/in/christina-brockway/[removed]" rel="nofollow noreferrer">
        <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
     </a> </li>
     <li><a href="https://github.com/dashboard" rel="nofollow noreferrer">
        <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
    </a></li>
</ul>"""
components.html(links_html)