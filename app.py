"""
Finley - AI Financial Advisor
Clean Chat Interface (Reference Screenshot Style)
"""

import streamlit as st
import sys
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from notebook code
from finley_core import (
    llm, app as workflow, conversation_manager,
    generate_allocation_pie_chart,
    generate_timeline_projection,
    compare_scenarios
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Finley - AI Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - CLEAN CHAT DESIGN
# ============================================================

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background - light */
    .main {
        background-color: #f5f5f5;
        padding: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Chat container - centered */
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 70%;
        text-align: left;
        display: block;
        word-wrap: break-word;
    }
    
    .bot-message {
        background-color: #e8eaf6;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 70%;
        display: block;
        word-wrap: break-word;
    }
    
    /* Info card in sidebar */
    .info-card {
        background-color: #34495e;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-label {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    
    .info-value {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Allocation card in sidebar */
    .allocation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .allocation-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .allocation-item {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    /* Input area */
    .stTextInput input {
        border-radius: 24px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Send button */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 24px;
        border: none;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Quick action buttons */
    div[data-testid="column"] button {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 500;
    }
    
    div[data-testid="column"] button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Chat history items */
    .chat-history-item {
        background-color: #34495e;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chat-history-item:hover {
        background-color: #435d7d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "age": None,
        "risk": None,
        "goal": None,
        "timeline": None,
        "emergency_fund": None,
        "debt": None
    }
    
if "allocation" not in st.session_state:
    st.session_state.allocation = None

if "timeline_data" not in st.session_state:
    st.session_state.timeline_data = None

if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = None

# ============================================================
# MESSAGE HANDLING FUNCTION
# ============================================================

def send_message(user_input: str):
    """Send message to Finley and get response"""
    if not user_input.strip():
        return
    
    # Import security check and patterns FIRST
    from finley_core import INJECTION_PATTERNS, SUSPICIOUS_KEYWORDS
    import re
    
    # SECURITY CHECK BEFORE DOING ANYTHING
    message_lower = user_input.lower()
    detected_issue = None
    
    # Check 1: Injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            detected_issue = f"injection pattern: {pattern}"
            break
    
    # Check 2: Suspicious keywords (creative content requests)
    if not detected_issue:
        suspicious_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw.lower() in message_lower)
        creative_keywords = ["poem", "story", "joke", "creative writing", "fiction"]
        has_creative_request = any(kw in message_lower for kw in creative_keywords)
        
        if has_creative_request:
            detected_issue = "creative content request (not financial advice)"
    
    # Check 3: Off-topic questions (NEW!)
    if not detected_issue:
        # Financial phrases (check these first, as two words)
        financial_phrases = [
            "stock market", "bond market", "financial market", "money market",
            "mutual fund", "index fund", "hedge fund", "exchange traded",
            "real estate", "interest rate", "credit card", "credit score",
            "emergency fund", "high interest", "net worth", "cash flow",
            "tax bracket", "capital gain", "dollar cost", "time horizon"
        ]
        
        # Financial keywords (single words)
        financial_keywords = [
            "invest", "money", "stock", "bond", "portfolio", "retirement",
            "saving", "save", "budget", "financial", "allocation", "allocate", 
            "risk", "401k", "roth", "ira", "fund", "market", "percent", 
            "cash", "debt", "emergency", "asset", "diversif", "compound", 
            "growth", "wealth", "income", "expense", "interest", "loan",
            "credit", "bank", "brokerage", "etf", "index", "mutual",
            "dividend", "tax", "inflation", "dollar", "euro", "currency",
            "finance", "advisor", "finley", "age", "goal", "timeline",
            "payment", "mortgage", "insurance", "trading", "crypto",
            "bitcoin", "coin", "yield", "return", "capital", "equity",
            "price", "value", "account", "contribution", "withdraw",
            "balance", "rate", "annual", "monthly", "fee", "cost"
        ]
        
        # Check if it's a question or request
        is_question = ("?" in user_input or 
                      message_lower.startswith(("what", "how", "why", "when", "who", 
                                                "where", "which", "tell me", "explain",
                                                "can you", "could you", "give me", "show me")))
        
        # Check financial phrases first (two-word combinations)
        has_financial_phrase = any(phrase in message_lower for phrase in financial_phrases)
        
        # Then check individual keywords
        has_financial_keyword = any(keyword in message_lower for keyword in financial_keywords)
        
        # If question with NO financial keywords/phrases AND longer than 3 words → likely off-topic
        if is_question and not (has_financial_phrase or has_financial_keyword) and len(user_input.split()) > 3:
            detected_issue = "off-topic question (not financial)"
    
    if detected_issue:
        # Security blocked - add both user message and security response
        st.session_state.messages.append({"role": "user", "content": user_input})
        security_response = "🛡️ I'm Finley, your AI financial advisor. I specialize in investment planning, retirement, and portfolio guidance. I can only help with financial questions. What would you like to know about investing or financial planning?"
        st.session_state.messages.append({"role": "assistant", "content": security_response})
        st.error(f"⚠️ Security Blocked: {detected_issue}")
        return  # STOP HERE - don't call workflow
    
    # Add user message to session (only if security passed)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from workflow
    try:
        # Use a unique thread_id to prevent LangGraph from loading old history
        # We manage conversation state in Streamlit, not in LangGraph's memory
        unique_thread = f"{st.session_state.thread_id}_{len(st.session_state.messages)}"
        config = {"configurable": {"thread_id": unique_thread}}
        
        # Convert session messages to LangChain format
        lc_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
        
        result = workflow.invoke({
            "user_input": user_input,
            "messages": lc_messages,
            "full_messages": lc_messages,
            "should_end": False,
            "needs_clarification": False,
            "response": "",
            "intent": "",
            "user_age": st.session_state.user_info.get("age"),
            "risk_tolerance": st.session_state.user_info.get("risk"),
            "investment_goal": st.session_state.user_info.get("goal"),
            "time_horizon": st.session_state.user_info.get("timeline"),
            "has_emergency_fund": st.session_state.user_info.get("emergency_fund"),
            "has_high_interest_debt": st.session_state.user_info.get("debt"),
            "job_stable": None,
            "educational_summary": None,
            "allocation_dict": st.session_state.allocation,
            "timeline_data": st.session_state.timeline_data,
            "scenario_data": st.session_state.scenario_data,
        }, config)
        
        # Extract response
        response = result.get("response", "")
        
        # Update session state with extracted info
        if result.get("user_age"):
            st.session_state.user_info["age"] = result["user_age"]
        if result.get("risk_tolerance"):
            st.session_state.user_info["risk"] = result["risk_tolerance"]
        if result.get("investment_goal"):
            st.session_state.user_info["goal"] = result["investment_goal"]
        if result.get("time_horizon"):
            st.session_state.user_info["timeline"] = result["time_horizon"]
        if result.get("has_emergency_fund") is not None:
            st.session_state.user_info["emergency_fund"] = result["has_emergency_fund"]
        if result.get("has_high_interest_debt") is not None:
            st.session_state.user_info["debt"] = result["has_high_interest_debt"]
        
        # Update allocation if generated
        if result.get("allocation_dict"):
            st.session_state.allocation = result["allocation_dict"]
        if result.get("timeline_data"):
            st.session_state.timeline_data = result["timeline_data"]
        if result.get("scenario_data"):
            st.session_state.scenario_data = result["scenario_data"]
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ============================================================
# SIDEBAR - USER INFO & ALLOCATION
# ============================================================

with st.sidebar:
    st.markdown("### 💰 Finley")
    st.markdown("Your AI Financial Advisor")
    st.markdown("---")
    
    # User Information Card
    st.markdown("#### 📊 Your Profile")
    
    for label, key, default in [
        ("Age", "age", "Not set"),
        ("Risk Tolerance", "risk", "Not set"),
        ("Goal", "goal", "Not set"),
        ("Timeline", "timeline", "Not set")
    ]:
        value = st.session_state.user_info.get(key)
        if value:
            display_value = f"{value} years" if key == "timeline" else str(value).title()
        else:
            display_value = default
        
        # Use simple markdown instead of HTML
        st.markdown(f"**{label}:** {display_value}")
    
    # Allocation Card (if generated)
    if st.session_state.allocation:
        st.markdown("---")
        st.markdown("#### 📊 Your Allocation")
        
        alloc = st.session_state.allocation
        
        # Use native Streamlit container with metrics
        with st.container():
            st.markdown("**Personalized Allocation**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stocks", f"{alloc.get('stocks', 0)}%")
            with col2:
                st.metric("Bonds", f"{alloc.get('bonds', 0)}%")
            with col3:
                st.metric("Cash", f"{alloc.get('cash', 0)}%")
        
        # Display charts in sidebar
        st.markdown("##### 📈 Visualizations")
        
        # Debug info
        chart_status = []
        if st.session_state.timeline_data:
            chart_status.append("Timeline ✅")
        if st.session_state.scenario_data:
            chart_status.append("Scenario ✅")
        
        if chart_status:
            st.markdown(f"*Available: {', '.join(chart_status)}*")
        
        # Pie chart
        try:
            import matplotlib.pyplot as plt
            plt.clf()
            generate_allocation_pie_chart(alloc)
            fig = plt.gcf()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Chart error: {e}")
        
        # Timeline projection (if available)
        if st.session_state.timeline_data:
            try:
                plt.clf()
                data = st.session_state.timeline_data
                generate_timeline_projection(
                    data['initial'],
                    data['monthly'],
                    data['years']
                )
                fig = plt.gcf()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Timeline error: {e}")
        
        # Scenario comparison (if available)
        if st.session_state.scenario_data:
            try:
                plt.clf()
                compare_scenarios(st.session_state.scenario_data)
                fig = plt.gcf()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Scenario error: {e}")
    
    st.markdown("---")
    
    # Save Current Conversation
    if st.session_state.messages:
        if st.button("💾 Save Current Chat", use_container_width=True):
            try:
                # Convert session messages to LangChain format for saving
                full_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        full_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        full_messages.append(AIMessage(content=msg["content"]))
                
                # Debug: Check what we're about to save
                has_timeline = st.session_state.timeline_data is not None
                has_scenario = st.session_state.scenario_data is not None
                
                # Save with proper state
                save_state = {
                    "messages": [],  # Not used, but needed for compatibility
                    "full_messages": full_messages,
                    "user_age": st.session_state.user_info.get("age"),
                    "risk_tolerance": st.session_state.user_info.get("risk"),
                    "investment_goal": st.session_state.user_info.get("goal"),
                    "time_horizon": st.session_state.user_info.get("timeline"),
                    "has_emergency_fund": st.session_state.user_info.get("emergency_fund"),
                    "has_high_interest_debt": st.session_state.user_info.get("debt"),
                    "allocation_dict": st.session_state.allocation,
                    "timeline_data": st.session_state.timeline_data,
                    "scenario_data": st.session_state.scenario_data,
                }
                
                filepath = conversation_manager.save_conversation(
                    st.session_state.thread_id, 
                    save_state
                )
                
                # Show what was saved
                saved_info = f"✅ Saved {len(st.session_state.messages)} messages"
                if has_timeline and has_scenario:
                    saved_info += " | All charts saved ✅"
                elif has_timeline or has_scenario:
                    saved_info += f" | ⚠️ Partial charts (timeline: {has_timeline}, scenario: {has_scenario})"
                else:
                    saved_info += " | ⚠️ No chart data to save"
                    
                st.success(saved_info)
                st.info(f"File: {filepath}")
            except Exception as e:
                st.error(f"Error saving: {e}")
    
    st.markdown("---")
    
    # Chat History
    st.markdown("#### 📜 History")
    
    try:
        saved_chats = conversation_manager.list_conversations()
        if saved_chats:
            # Sort by timestamp, most recent first
            saved_chats = sorted(saved_chats, key=lambda x: x['timestamp'], reverse=True)
            
            for chat_info in saved_chats[:5]:  # Show last 5
                chat_id = chat_info['thread_id']
                msg_count = chat_info['message_count']
                timestamp = chat_info['timestamp'][:10]  # Just the date
                
                button_label = f"💬 {timestamp} ({msg_count} msgs)"
                
                if st.button(button_label, key=f"load_{chat_id}", use_container_width=True):
                    # Load conversation
                    loaded = conversation_manager.load_conversation(chat_id)
                    if loaded:
                        # IMPORTANT: Clear current messages first
                        st.session_state.messages = []
                        
                        # Convert LangChain messages to session format
                        for msg in loaded.get('full_messages', []):
                            if isinstance(msg, HumanMessage):
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": msg.content
                                })
                            elif isinstance(msg, AIMessage):
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": msg.content
                                })
                        
                        # Restore user info
                        st.session_state.user_info = {
                            'age': loaded.get('user_age'),
                            'risk': loaded.get('risk_tolerance'),
                            'goal': loaded.get('investment_goal'),
                            'timeline': loaded.get('time_horizon'),
                            'emergency_fund': loaded.get('has_emergency_fund'),
                            'debt': loaded.get('has_high_interest_debt')
                        }
                        
                        # Restore allocation and chart data
                        st.session_state.allocation = loaded.get('allocation_dict')
                        st.session_state.timeline_data = loaded.get('timeline_data')
                        st.session_state.scenario_data = loaded.get('scenario_data')
                        
                        # Debug: Show what was loaded
                        debug_info = []
                        if st.session_state.allocation:
                            debug_info.append("✅ Allocation")
                        if st.session_state.timeline_data:
                            debug_info.append("✅ Timeline")
                        if st.session_state.scenario_data:
                            debug_info.append("✅ Scenario")
                        
                        # IMPORTANT: Update thread_id to continue in this conversation
                        st.session_state.thread_id = chat_id
                        
                        loaded_msg = f"✅ Loaded: {len(st.session_state.messages)} messages"
                        if debug_info:
                            loaded_msg += f" | Charts: {', '.join(debug_info)}"
                        else:
                            loaded_msg += " | ⚠️ No chart data saved"
                        
                        st.success(loaded_msg)
                        st.rerun()
        else:
            st.markdown("*No saved chats*")
    except Exception as e:
        st.markdown(f"*Error loading chats: {e}*")
    
    st.markdown("---")
    
    # New Chat Button
    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.user_info = {
            "age": None, "risk": None, "goal": None, "timeline": None,
            "emergency_fund": None, "debt": None
        }
        st.session_state.allocation = None
        st.session_state.timeline_data = None
        st.session_state.scenario_data = None
        st.rerun()
    
    # Clear All History Button
    if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
        import os
        import glob
        conv_dir = "finley_conversations"
        if os.path.exists(conv_dir):
            files = glob.glob(os.path.join(conv_dir, "*.json"))
            deleted = 0
            for f in files:
                try:
                    os.remove(f)
                    deleted += 1
                except:
                    pass
            if deleted > 0:
                st.success(f"✅ Deleted {deleted} conversations")
                st.rerun()
            else:
                st.info("No conversations to clear")
        else:
            st.info("No conversations to clear")
    
    st.markdown("---")
    st.markdown("##### 👥 Team")
    st.markdown("""
    - Aditya Pimpalkar
    - Diksha Sahare
    - Harsh Jatin Patel
    - Tapan Chandrakant Patel
    """)
    st.markdown("**INFO 7375** - Prompt Engineering")
    st.markdown("*Prof. Shirali Patel*")

# ============================================================
# MAIN CHAT AREA
# ============================================================

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header
st.markdown("### 💰 Finley - Your AI Financial Advisor")
if st.session_state.messages:
    msg_count = len(st.session_state.messages)
    st.markdown(f"*Chat: {st.session_state.thread_id} ({msg_count} messages)*")
st.markdown("---")

# Display chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    
    if role == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 8px 0;">
            <div class="user-message">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 8px 0;">
            <div class="bot-message">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# Show Finley introduction if no messages
if not st.session_state.messages:
    # Welcome header
    st.markdown("""
    <div style="padding: 2rem 1rem; text-align: center;">
        <h2 style="color: #667eea; margin-bottom: 1.5rem;">👋 Welcome to Finley!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Hi! I'm **Finley**, your AI-powered financial advisor. I'm here to help you make informed 
    investment decisions through personalized guidance tailored to your unique financial situation.
    """)
    
    st.markdown("---")
    
    # What Finley can do
    st.markdown("### 🎯 What I Can Help You With:")
    
    st.markdown("""
    - 💰 **Personalized Investment Portfolios** - Based on your age, goals, and risk tolerance
    - 📊 **Risk Assessment** - Understanding your comfort level with market volatility
    - 🎓 **Financial Education** - Learn about stocks, bonds, diversification, and more
    - 📈 **Retirement Planning** - Calculate what you need to retire comfortably
    - 🔮 **Scenario Analysis** - See how different strategies might perform
    """)
    
    st.markdown("---")
    
    # Call to action
    st.info("""
    🚀 **Let's Get Started!**
    
    I'll ask you a few questions about your financial situation to create a personalized investment 
    strategy. Don't worry - all your information stays private and secure.
    """)
    
    # Team credits
    st.markdown("""
    <p style="text-align: center; margin-top: 2rem; color: #888; font-size: 0.9rem;">
        <em>Built by Team Finley: Aditya, Diksha, Harsh, and Tapan</em><br>
        <em>INFO 7375 - Prompt Engineering for Generative AI | Northeastern University</em>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick action buttons below intro
    st.markdown("### Quick Actions")
    st.markdown("Choose how you'd like to start:")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Get Started", use_container_width=True, key="quick_start"):
            send_message("Help me invest")
            st.rerun()
    
    with col2:
        if st.button("💡 Learn Investing", use_container_width=True, key="quick_learn"):
            send_message("Teach me about investing")
            st.rerun()
    
    with col3:
        if st.button("❓ Ask Question", use_container_width=True, key="quick_question"):
            send_message("I have a question about investing")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INPUT AREA
# ============================================================

# Initialize input counter for clearing
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# Create columns for input and send button
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_counter}"  # Dynamic key clears input
    )

with col2:
    send_button = st.button("Send", use_container_width=True)

# Handle send button click or Enter key
if send_button and user_input:
    send_message(user_input)
    st.session_state.input_counter += 1  # Increment to clear input
    st.rerun()  # Refresh to show new messages

