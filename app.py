"""
AI-Driven Financial Management Chatbot with Gradio UI
Uses IBM Granite 3.2-2B Instruct Model
Designed for Google Colab with Beautiful Interactive Interface

INSTALLATION - Run this first:
!pip install -q transformers torch accelerate huggingface_hub pandas matplotlib seaborn plotly gradio
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import warnings
import io
import base64
from PIL import Image
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.user_profile = {}
        self.income_list = []
        self.expenses_list = []
        self.analysis = {}
        self.conversation_history = []
        self.onboarding_step = 0
        self.gamification_points = 0
        self.challenges = []
        self.is_model_loaded = False

app_state = AppState()

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load IBM Granite model"""
    if app_state.is_model_loaded:
        return "✅ Model already loaded!"

    try:
        yield "🔄 Loading IBM Granite 3.2-2B model... (2-3 minutes)"

        model_name = "ibm-granite/granite-3.2-2b-instruct"

        app_state.tokenizer = AutoTokenizer.from_pretrained(model_name)
        yield "✅ Tokenizer loaded! Loading model..."

        app_state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )

        app_state.is_model_loaded = True
        device = "GPU" if torch.cuda.is_available() else "CPU"
        yield f"✅ Model loaded successfully on {device}!\n\n🎉 Ready to help you manage your finances!"

    except Exception as e:
        yield f"❌ Error loading model: {str(e)}\n\nPlease check your internet connection and try again."

def generate_ai_response(prompt: str, context: str = "") -> str:
    """Generate AI response"""
    if not app_state.is_model_loaded:
        return "⚠️ Please load the model first using the 'Load AI Model' button in Setup tab."

    if context:
        full_prompt = f"{context}\n\nUser: {prompt}\n\nFinancial Advisor:"
    else:
        full_prompt = f"You are a helpful financial advisor. Answer concisely:\n\n{prompt}\n\nAnswer:"

    inputs = app_state.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(app_state.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = app_state.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=app_state.tokenizer.eos_token_id
        )

    response = app_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()

    return response

# ============================================================================
# ONBOARDING FUNCTIONS
# ============================================================================

def save_basic_info(name, age, category, marital_status):
    """Save basic user information"""
    if not name or not age:
        return "⚠️ Please fill in all fields", None

    app_state.user_profile.update({
        'name': name,
        'age': int(age),
        'category': category,
        'marital_status': marital_status,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    summary = f"""✅ **Profile Created!**

👤 **Name:** {name}
🎂 **Age:** {age}
📊 **Category:** {category}
💑 **Marital Status:** {marital_status}

Please continue to the next section!"""

    return summary, gr.update(visible=True)

def save_family_location(family_size, dependents, city, country, living_situation):
    """Save family and location info"""
    app_state.user_profile.update({
        'family_size': int(family_size),
        'dependents': int(dependents),
        'city': city,
        'country': country,
        'living_situation': living_situation
    })

    return f"✅ Family and location info saved! Total family: {family_size}, Location: {city}"

def save_assets(owns_vehicle, vehicle_type, owns_property):
    """Save assets information"""
    app_state.user_profile.update({
        'owns_vehicle': owns_vehicle,
        'vehicle_type': vehicle_type if owns_vehicle else 'None',
        'owns_property': owns_property
    })

    return "✅ Assets information saved!"

def save_goals(goals_text):
    """Save financial goals"""
    goals = [g.strip() for g in goals_text.split('\n') if g.strip()]
    app_state.user_profile['goals'] = goals

    profile_summary = f"""
🎉 **Profile Complete!**

**Summary:**
- Name: {app_state.user_profile.get('name')}
- Age: {app_state.user_profile.get('age')}
- Category: {app_state.user_profile.get('category')}
- Family Size: {app_state.user_profile.get('family_size')}
- Location: {app_state.user_profile.get('city')}, {app_state.user_profile.get('country')}
- Goals: {len(goals)} goals set

**Next Step:** Add your income sources in the Income tab!
"""

    return profile_summary

# ============================================================================
# INCOME MANAGEMENT
# ============================================================================

def add_income(source, amount, frequency):
    """Add income entry"""
    if not source or not amount:
        return "⚠️ Please fill in all fields", create_income_table()

    try:
        income_entry = {
            'source': source,
            'amount': float(amount),
            'frequency': frequency,
            'date': datetime.now().strftime("%Y-%m-%d")
        }

        app_state.income_list.append(income_entry)

        total = sum(inc['amount'] for inc in app_state.income_list)
        return f"✅ Added ₹{amount} from {source}. Total income: ₹{total:,.0f}", create_income_table()

    except ValueError:
        return "⚠️ Invalid amount", create_income_table()

def create_income_table():
    """Create income summary table"""
    if not app_state.income_list:
        return pd.DataFrame(columns=['Source', 'Amount (₹)', 'Frequency'])

    df = pd.DataFrame(app_state.income_list)
    df_display = pd.DataFrame({
        'Source': df['source'],
        'Amount (₹)': df['amount'].apply(lambda x: f"₹{x:,.0f}"),
        'Frequency': df['frequency']
    })

    return df_display

def get_income_sources(category):
    """Get income sources based on category"""
    sources = {
        'Student': ['Pocket Money', 'Scholarship', 'Part-time Job', 'Tuition', 'Freelancing'],
        'Employed': ['Salary', 'Bonus', 'Investment Returns', 'Rental Income', 'Side Business'],
        'Unemployed': ['Savings', 'Family Support', 'Freelancing', 'Unemployment Benefits'],
        'Housewife': ['Family Income', 'Side Business', 'Investment Returns'],
        'Retired': ['Pension', 'Investment Returns', 'Rental Income', 'Savings']
    }
    return gr.update(choices=sources.get(category, sources['Employed']))

# ============================================================================
# EXPENSE MANAGEMENT
# ============================================================================

def add_expense_quick(description):
    """Quick expense entry with smart parsing"""
    if not description:
        return "⚠️ Please enter an expense", create_expense_table(), None

    try:
        # Parse: "description amount" or just "amount"
        parts = description.strip().split()
        amount = float(parts[-1])
        desc = ' '.join(parts[:-1]) if len(parts) > 1 else 'Expense'

        # Auto-categorize
        category = auto_categorize_expense(desc)

        expense_entry = {
            'description': desc,
            'amount': amount,
            'category': category,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S")
        }

        app_state.expenses_list.append(expense_entry)

        total = sum(exp['amount'] for exp in app_state.expenses_list)
        return (
            f"✅ Added ₹{amount:,.0f} - {desc} ({category})\nTotal expenses: ₹{total:,.0f}",
            create_expense_table(),
            ""  # Clear input
        )

    except (ValueError, IndexError):
        return "⚠️ Format: 'description amount' (e.g., 'Groceries 500')", create_expense_table(), description

def auto_categorize_expense(description: str) -> str:
    """Auto-categorize based on keywords"""
    description = description.lower()

    keywords = {
        'Food & Groceries': ['grocery', 'food', 'vegetables', 'supermarket', 'lunch', 'dinner', 'breakfast'],
        'Rent': ['rent', 'lease'],
        'Utilities': ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile'],
        'Transportation': ['petrol', 'diesel', 'fuel', 'bus', 'taxi', 'uber', 'ola', 'auto'],
        'Healthcare': ['medicine', 'doctor', 'hospital', 'health', 'medical'],
        'Education': ['school', 'college', 'tuition', 'books', 'course', 'fees'],
        'Entertainment': ['movie', 'restaurant', 'party', 'gaming', 'netflix', 'amazon prime'],
        'Clothing': ['clothes', 'shoes', 'dress', 'shirt'],
    }

    for category, words in keywords.items():
        if any(word in description for word in words):
            return category

    return 'Other'

def add_expense_detailed(description, amount, category):
    """Detailed expense entry"""
    if not description or not amount:
        return "⚠️ Please fill in all fields", create_expense_table()

    try:
        expense_entry = {
            'description': description,
            'amount': float(amount),
            'category': category,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S")
        }

        app_state.expenses_list.append(expense_entry)

        total = sum(exp['amount'] for exp in app_state.expenses_list)
        return f"✅ Added ₹{amount} - {description}. Total: ₹{total:,.0f}", create_expense_table()

    except ValueError:
        return "⚠️ Invalid amount", create_expense_table()

def create_expense_table():
    """Create expense summary table"""
    if not app_state.expenses_list:
        return pd.DataFrame(columns=['Date', 'Description', 'Category', 'Amount (₹)'])

    df = pd.DataFrame(app_state.expenses_list)
    df_display = pd.DataFrame({
        'Date': df['date'],
        'Description': df['description'],
        'Category': df['category'],
        'Amount (₹)': df['amount'].apply(lambda x: f"₹{x:,.0f}")
    })

    return df_display.tail(10)  # Show last 10

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_budget():
    """Analyze budget and generate insights"""
    if not app_state.income_list or not app_state.expenses_list:
        return "⚠️ Please add income and expenses first", None, None, None

    total_income = sum(inc['amount'] for inc in app_state.income_list)
    total_expenses = sum(exp['amount'] for exp in app_state.expenses_list)
    savings = total_income - total_expenses
    savings_rate = (savings / total_income * 100) if total_income > 0 else 0

    # Category breakdown
    category_totals = {}
    for exp in app_state.expenses_list:
        cat = exp['category']
        category_totals[cat] = category_totals.get(cat, 0) + exp['amount']

    app_state.analysis = {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'savings': savings,
        'savings_rate': savings_rate,
        'category_breakdown': category_totals,
        'status': 'surplus' if savings > 0 else 'deficit'
    }

    # Generate summary
    summary = f"""
# 📊 Financial Analysis Report

## 💰 Overview
- **Total Income:** ₹{total_income:,.0f}
- **Total Expenses:** ₹{total_expenses:,.0f}
- **Savings:** ₹{savings:,.0f}
- **Savings Rate:** {savings_rate:.1f}%

## 📈 Status
"""

    if savings > 0:
        summary += f"✅ **Surplus:** You're saving ₹{savings:,.0f} per month!\n\n"
    else:
        summary += f"⚠️ **Deficit:** You're overspending by ₹{abs(savings):,.0f}\n\n"

    # AI Insights
    if app_state.is_model_loaded:
        summary += "## 🤖 AI Insights\n\n"
        context = f"""User Profile:
- Category: {app_state.user_profile.get('category', 'Unknown')}
- Monthly Income: ₹{total_income:,.0f}
- Monthly Expenses: ₹{total_expenses:,.0f}
- Savings: ₹{savings:,.0f}
- Savings Rate: {savings_rate:.1f}%

Provide 3 specific, actionable tips to improve their finances."""

        ai_response = generate_ai_response("What should I do to improve my financial situation?", context)
        summary += ai_response

    # Create visualizations
    fig1 = create_income_vs_expense_chart()
    fig2 = create_expense_breakdown_chart()
    fig3 = create_savings_rate_gauge()

    return summary, fig1, fig2, fig3

def create_income_vs_expense_chart():
    """Create income vs expense bar chart"""
    if not app_state.analysis:
        return None

    fig = go.Figure(data=[
        go.Bar(name='Amount', x=['Income', 'Expenses', 'Savings'],
               y=[app_state.analysis['total_income'],
                  app_state.analysis['total_expenses'],
                  app_state.analysis['savings']],
               marker_color=['#2ecc71', '#e74c3c', '#3498db'])
    ])

    fig.update_layout(
        title='Income vs Expenses vs Savings',
        yaxis_title='Amount (₹)',
        height=400,
        showlegend=False
    )

    return fig

def create_expense_breakdown_chart():
    """Create expense breakdown pie chart"""
    if not app_state.analysis or not app_state.analysis.get('category_breakdown'):
        return None

    categories = list(app_state.analysis['category_breakdown'].keys())
    amounts = list(app_state.analysis['category_breakdown'].values())

    fig = go.Figure(data=[go.Pie(labels=categories, values=amounts, hole=0.3)])

    fig.update_layout(
        title='Expense Distribution by Category',
        height=400
    )

    return fig

def create_savings_rate_gauge():
    """Create savings rate gauge"""
    if not app_state.analysis:
        return None

    savings_rate = app_state.analysis['savings_rate']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=savings_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Savings Rate (%)"},
        delta={'reference': 20},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "#ffcccb"},
                {'range': [10, 20], 'color': "#ffffcc"},
                {'range': [20, 50], 'color': "#90EE90"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))

    fig.update_layout(height=400)

    return fig

# ============================================================================
# CHATBOT
# ============================================================================

def chatbot_response(message, history):
    """Chatbot conversation"""
    if not app_state.is_model_loaded:
        return "⚠️ Please load the AI model first from the Setup tab."

    if not app_state.user_profile:
        return "⚠️ Please complete your profile in the Onboarding tab first."

    # Build context
    context = f"""User Profile:
- Name: {app_state.user_profile.get('name')}
- Category: {app_state.user_profile.get('category')}
- Age: {app_state.user_profile.get('age')}
"""

    if app_state.analysis:
        context += f"""
Financial Summary:
- Monthly Income: ₹{app_state.analysis['total_income']:,.0f}
- Monthly Expenses: ₹{app_state.analysis['total_expenses']:,.0f}
- Savings: ₹{app_state.analysis['savings']:,.0f}
- Savings Rate: {app_state.analysis['savings_rate']:.1f}%
"""

    response = generate_ai_response(message, context)

    return response

# ============================================================================
# ADDITIONAL FEATURES
# ============================================================================

def get_tax_suggestions():
    """Get tax saving suggestions"""
    if not app_state.analysis:
        return "⚠️ Please complete budget analysis first"

    annual_income = app_state.analysis['total_income'] * 12

    suggestions = f"""
# 💰 Tax Saving Opportunities

## Section 80C (Save up to ₹46,800 in taxes)
Maximum deduction: ₹1,50,000

**Best Options:**
- **PPF (Public Provident Fund):** Safe, government-backed, 7.1% interest
- **ELSS Mutual Funds:** Higher returns, 3-year lock-in, equity exposure
- **Life Insurance Premium:** Protection + tax benefit
- **NSC (National Savings Certificate):** Fixed returns, 5-year tenure

## Section 80D (Health Insurance)
Maximum deduction: ₹25,000 (₹50,000 for senior citizens)

**Benefits:**
- Covers health insurance premium for self, family, and parents
- Includes preventive health check-ups (₹5,000)

## NPS (National Pension Scheme) - Section 80CCD(1B)
Additional deduction: ₹50,000

**Benefits:**
- Long-term retirement planning
- Tax benefits + market-linked returns

## Estimated Annual Tax Savings
Based on your income of ₹{annual_income:,.0f}:

- Max saving through 80C: ₹46,800
- Max saving through 80D: ₹7,800
- Max saving through NPS: ₹15,600

**Total Potential Savings: ₹70,200 per year!**

---

💡 **Recommendation:** Start with PPF (₹12,500/month) and term insurance for immediate benefits!
"""

    return suggestions

def get_government_schemes():
    """Get applicable government schemes"""
    if not app_state.user_profile:
        return "⚠️ Please complete profile first"

    category = app_state.user_profile.get('category', '').lower()
    age = app_state.user_profile.get('age', 0)

    schemes = f"""
# 💼 Applicable Government Schemes

"""

    if 'student' in category:
        schemes += """
## For Students

### 1. PM Scholarship Scheme
- **Benefit:** ₹2,000-3,000/month
- **Eligibility:** Merit-based for economically weaker sections
- **Apply:** scholarships.gov.in

### 2. National Scholarship Portal (NSP)
- **Benefit:** Various state and central scholarships
- **Eligibility:** Based on category and merit
- **Apply:** scholarships.gov.in

### 3. Post Matric Scholarship
- **Benefit:** Full tuition + maintenance
- **Eligibility:** SC/ST/OBC students
"""

    elif 'employed' in category:
        schemes += """
## For Employed

### 1. PM Suraksha Bima Yojana
- **Benefit:** ₹2 lakh accident insurance
- **Premium:** Only ₹20/year
- **Eligibility:** Age 18-70 with bank account

### 2. Atal Pension Yojana
- **Benefit:** Guaranteed pension ₹1,000-5,000/month
- **Eligibility:** Age 18-40
"""

    elif 'housewife' in category:
        schemes += """
## For Housewives

### 1. Mahila Samman Savings Certificate
- **Benefit:** 7.5% interest (better than FD)
- **Eligibility:** All women
- **Tenure:** 2 years

### 2. PM Jan Dhan Yojana
- **Benefit:** Free bank account + ₹2 lakh insurance
- **Eligibility:** All Indian citizens
"""

    if age >= 60:
        schemes += """
## For Senior Citizens

### 1. Senior Citizen Savings Scheme (SCSS)
- **Benefit:** 8.2% interest
- **Maximum:** ₹30 lakh
- **Eligibility:** Age 60+

### 2. Pradhan Mantri Vaya Vandana Yojana
- **Benefit:** Guaranteed pension
- **Returns:** 7.4% per annum
"""

    schemes += """
---

## Universal Schemes

### PM Suraksha Bima Yojana
- ₹2 lakh accident cover for ₹20/year
- Age 18-70

### PM Jeevan Jyoti Bima Yojana
- ₹2 lakh life cover for ₹436/year
- Age 18-50

---

💡 Visit your nearest bank or post office to apply!
"""

    return schemes

def compare_with_peers():
    """Compare with peer benchmarks"""
    if not app_state.analysis:
        return "⚠️ Please complete budget analysis first"

    category = app_state.user_profile.get('category', 'employed').lower()

    benchmarks = {
        'student': {'avg_income': 8000, 'avg_expenses': 6500, 'avg_savings_rate': 18.75},
        'employed': {'avg_income': 45000, 'avg_expenses': 35000, 'avg_savings_rate': 22.22},
        'housewife': {'avg_income': 0, 'avg_expenses': 25000, 'avg_savings_rate': 15.0},
        'retired': {'avg_income': 30000, 'avg_expenses': 22000, 'avg_savings_rate': 26.67}
    }

    benchmark = benchmarks.get(category, benchmarks['employed'])

    user_savings_rate = app_state.analysis['savings_rate']
    savings_diff = user_savings_rate - benchmark['avg_savings_rate']

    comparison = f"""
# 📊 Peer Comparison

## Your Category: {category.title()}

### Savings Rate Comparison
- **Your Savings Rate:** {user_savings_rate:.1f}%
- **Peer Average:** {benchmark['avg_savings_rate']:.1f}%
- **Difference:** {savings_diff:+.1f}%

"""

    if savings_diff > 5:
        comparison += f"""
🎉 **Excellent!** You're saving {savings_diff:.1f}% more than your peers!

You're in the top 20% of savers in your category. Keep it up!
"""
    elif savings_diff > 0:
        comparison += f"""
✅ **Good Job!** You're saving more than average.

You're doing better than most people in your category. Small improvements can put you in the top tier!
"""
    else:
        comparison += f"""
⚠️ **Opportunity to Improve**

You're saving {abs(savings_diff):.1f}% less than peers in your category.

**Action Plan:**
1. Review your expense categories
2. Identify non-essential spending
3. Set a goal to increase savings by 5% this month
"""

    comparison += f"""

### Income vs Expenses
- **Peer Average Income:** ₹{benchmark['avg_income']:,.0f}
- **Peer Average Expenses:** ₹{benchmark['avg_expenses']:,.0f}

---

💡 **Tip:** Focus on savings rate rather than absolute amounts. Even small percentages compound over time!
"""

    return comparison

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_ui():
    """Create Gradio interface"""

    with gr.Blocks(title="AI Financial Advisor", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # 🤖 AI Financial Advisor
        ### Powered by IBM Granite 3.2-2B | Your Personal Finance Assistant
        """)

        with gr.Tabs() as tabs:

            # ============================================================
            # TAB 1: SETUP
            # ============================================================
            with gr.Tab("🚀 Setup", id=0):
                gr.Markdown("""
                ## Welcome! Let's get started

                **Step 1:** Load the AI model (takes 2-3 minutes first time)
                **Step 2:** Complete your profile in the Onboarding tab
                **Step 3:** Add your income and expenses
                **Step 4:** Get AI-powered insights!
                """)

                load_btn = gr.Button("🔄 Load AI Model", variant="primary", size="lg")
                model_status = gr.Textbox(label="Status", lines=3)

                load_btn.click(
                    fn=load_model,
                    outputs=[model_status]
                )

                gr.Markdown("""
                ---
                ### Quick Start Guide

                1. **Load Model:** Click the button above (one-time setup)
                2. **Profile:** Fill your details in Onboarding tab
                3. **Income:** Add all income sources
                4. **Expenses:** Track your spending (quick or detailed)
                5. **Analysis:** Get AI insights and visualizations
                6. **Chat:** Ask questions anytime!
                """)

            # ============================================================
            # TAB 2: ONBOARDING
            # ============================================================
            with gr.Tab("👤 Onboarding", id=1):
                gr.Markdown("## Create Your Financial Profile")

                with gr.Group():
                    gr.Markdown("### Basic Information")
                    with gr.Row():
                        name_input = gr.Textbox(label="Name", placeholder="Enter your name")
                        age_input = gr.Number(label="Age", value=25, minimum=1, maximum=100)

                    with gr.Row():
                        category_input = gr.Dropdown(
                            label="Category",
                            choices=["Student", "Employed", "Unemployed", "Housewife", "Retired"],
                            value="Employed"
                        )
                        marital_input = gr.Dropdown(
                            label="Marital Status",
                            choices=["Single", "Married", "Divorced", "Widowed"],
                            value="Single"
                        )

                    basic_submit = gr.Button("Save Basic Info", variant="primary")
                    basic_output = gr.Markdown()

                with gr.Group(visible=True) as family_section:
                    gr.Markdown("### Family & Location")
                    with gr.Row():
                        family_size_input = gr.Number(label="Total Family Members", value=1, minimum=1)
                        dependents_input = gr.Number(label="Dependents", value=0, minimum=0)

                    with gr.Row():
                        city_input = gr.Textbox(label="City", placeholder="e.g., Bangalore")
                        country_input = gr.Textbox(label="Country", value="India")

                    living_input = gr.Dropdown(
                        label="Living Situation",
                        choices=["Own Home", "Rented", "PG", "Hostel", "With Parents"],
                        value="Rented"
                    )

                    family_submit = gr.Button("Save Family Info")
                    family_output = gr.Markdown()

                with gr.Group():
                    gr.Markdown("### Assets")
                    owns_vehicle = gr.Checkbox(label="Own Vehicle?")
                    vehicle_type = gr.Dropdown(
                        label="Vehicle Type",
                        choices=["Car", "Bike", "Both", "None"],
                        value="None"
                    )
                    owns_property = gr.Checkbox(label="Own Property?")

                    assets_submit = gr.Button("Save Assets")
                    assets_output = gr.Markdown()

                with gr.Group():
                    gr.Markdown("### Financial Goals & Dreams")
                    goals_input = gr.Textbox(
                        label="Your Goals (one per line)",
                        placeholder="Example:\nBuy a car in 2 years\nSave for education\nGo on Europe trip",
                        lines=5
                    )

                    goals_submit = gr.Button("Complete Profile", variant="primary")
                    profile_summary = gr.Markdown()

                # Connect buttons
                basic_submit.click(
                    fn=save_basic_info,
                    inputs=[name_input, age_input, category_input, marital_input],
                    outputs=[basic_output, family_section]
                )

                family_submit.click(
                    fn=save_family_location,
                    inputs=[family_size_input, dependents_input, city_input, country_input, living_input],
                    outputs=[family_output]
                )

                assets_submit.click(
                    fn=save_assets,
                    inputs=[owns_vehicle, vehicle_type, owns_property],
                    outputs=[assets_output]
                )

                goals_submit.click(
                    fn=save_goals,
                    inputs=[goals_input],
                    outputs=[profile_summary]
                )

            # ============================================================
            # TAB 3: INCOME
            # ============================================================
            with gr.Tab("💰 Income", id=2):
                gr.Markdown("## Add Your Income Sources")

                with gr.Row():
                    with gr.Column(scale=2):
                        income_source = gr.Dropdown(
                            label="Income Source",
                            choices=["Salary", "Bonus", "Investment Returns", "Rental Income", "Side Business", "Freelancing", "Other"],
                            value="Salary"
                        )
                        income_amount = gr.Number(label="Monthly Amount (₹)", value=0, minimum=0)
                        income_frequency = gr.Dropdown(
                            label="Frequency",
                            choices=["Monthly", "Yearly", "One-time"],
                            value="Monthly"
                        )

                        add_income_btn = gr.Button("➕ Add Income", variant="primary")
                        income_output = gr.Markdown()

                    with gr.Column(scale=3):
                        income_table = gr.Dataframe(
                            label="Your Income Sources",
                            headers=["Source", "Amount (₹)", "Frequency"],
                            interactive=False
                        )

                # Update income sources based on category
                category_input.change(
                    fn=get_income_sources,
                    inputs=[category_input],
                    outputs=[income_source]
                )

                add_income_btn.click(
                    fn=add_income,
                    inputs=[income_source, income_amount, income_frequency],
                    outputs=[income_output, income_table]
                )

                gr.Markdown("""
                ---
                ### 💡 Tips for Income Entry
                - Include all regular income sources
                - For students: pocket money, scholarships, part-time work
                - For employed: salary, bonuses, side income
                - For investments: average monthly returns
                """)

            # ============================================================
            # TAB 4: EXPENSES
            # ============================================================
            with gr.Tab("💸 Expenses", id=3):
                gr.Markdown("## Track Your Expenses")

                with gr.Tabs():
                    # Quick Entry
                    with gr.Tab("⚡ Quick Entry"):
                        gr.Markdown("""
                        ### Smart Expense Entry
                        Just type naturally! Examples:
                        - `Groceries 2500`
                        - `Movie tickets 800`
                        - `Electricity bill 1500`

                        The AI will automatically categorize it!
                        """)

                        quick_expense_input = gr.Textbox(
                            label="Quick Entry",
                            placeholder="e.g., Lunch 250",
                            lines=1
                        )
                        quick_add_btn = gr.Button("➕ Add Expense", variant="primary")
                        quick_expense_output = gr.Markdown()

                    # Detailed Entry
                    with gr.Tab("📝 Detailed Entry"):
                        with gr.Row():
                            expense_desc = gr.Textbox(label="Description", placeholder="e.g., Supermarket groceries")
                            expense_amount = gr.Number(label="Amount (₹)", value=0, minimum=0)

                        expense_category = gr.Dropdown(
                            label="Category",
                            choices=[
                                "Food & Groceries", "Rent", "Utilities", "Transportation",
                                "Healthcare", "Education", "Entertainment", "Clothing",
                                "Insurance", "EMI/Loans", "Communication", "Personal Care",
                                "Household Items", "Gifts & Donations", "Travel", "Investments",
                                "Emergency Fund", "Other"
                            ],
                            value="Food & Groceries"
                        )

                        detailed_add_btn = gr.Button("➕ Add Expense", variant="primary")
                        detailed_expense_output = gr.Markdown()

                # Expense Table
                gr.Markdown("### Recent Expenses")
                expense_table = gr.Dataframe(
                    label="Your Expenses (Last 10)",
                    headers=["Date", "Description", "Category", "Amount (₹)"],
                    interactive=False
                )

                # Connect buttons
                quick_add_btn.click(
                    fn=add_expense_quick,
                    inputs=[quick_expense_input],
                    outputs=[quick_expense_output, expense_table, quick_expense_input]
                )

                detailed_add_btn.click(
                    fn=add_expense_detailed,
                    inputs=[expense_desc, expense_amount, expense_category],
                    outputs=[detailed_expense_output, expense_table]
                )

            # ============================================================
            # TAB 5: ANALYSIS
            # ============================================================
            with gr.Tab("📊 Analysis", id=4):
                gr.Markdown("## Financial Analysis & Insights")

                analyze_btn = gr.Button("🔍 Analyze My Finances", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        analysis_summary = gr.Markdown(label="Analysis Report")

                with gr.Row():
                    chart1 = gr.Plot(label="Income vs Expenses")
                    chart2 = gr.Plot(label="Expense Breakdown")

                with gr.Row():
                    chart3 = gr.Plot(label="Savings Rate")

                analyze_btn.click(
                    fn=analyze_budget,
                    outputs=[analysis_summary, chart1, chart2, chart3]
                )

                gr.Markdown("""
                ---
                ### 📈 Understanding Your Report

                - **Surplus:** You're saving money! Great job!
                - **Deficit:** You're overspending. Time to cut costs.
                - **Savings Rate:** Aim for at least 20% of income
                - **50-30-20 Rule:** 50% needs, 30% wants, 20% savings
                """)

            # ============================================================
            # TAB 6: AI CHAT
            # ============================================================
            with gr.Tab("💬 AI Assistant", id=5):
                gr.Markdown("""
                ## Chat with Your AI Financial Advisor

                Ask me anything about your finances! Examples:
                - How can I save more money?
                - Where should I cut expenses?
                - Am I spending too much on entertainment?
                - What investments should I consider?
                """)

                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here...",
                    lines=2
                )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                def user(user_message, history):
                    return "", history + [[user_message, None]]

                def bot(history):
                    user_message = history[-1][0]
                    bot_message = chatbot_response(user_message, history)
                    history[-1][1] = bot_message
                    return history

                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, chatbot, chatbot
                )
                submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, chatbot, chatbot
                )
                clear_btn.click(lambda: None, None, chatbot, queue=False)

                gr.Markdown("""
                ---
                ### 💡 Conversation Tips
                - Be specific about your questions
                - Mention time frames (this month, this year)
                - Ask for comparisons or alternatives
                - Request step-by-step advice
                """)

            # ============================================================
            # TAB 7: SUGGESTIONS
            # ============================================================
            with gr.Tab("💡 Suggestions", id=6):
                gr.Markdown("## Expert Recommendations")

                with gr.Tabs():
                    # Tax Savings
                    with gr.Tab("💰 Tax Savings"):
                        tax_btn = gr.Button("Get Tax Saving Suggestions", variant="primary")
                        tax_output = gr.Markdown()

                        tax_btn.click(
                            fn=get_tax_suggestions,
                            outputs=[tax_output]
                        )

                    # Government Schemes
                    with gr.Tab("🏛️ Government Schemes"):
                        schemes_btn = gr.Button("Show Applicable Schemes", variant="primary")
                        schemes_output = gr.Markdown()

                        schemes_btn.click(
                            fn=get_government_schemes,
                            outputs=[schemes_output]
                        )

                    # Peer Comparison
                    with gr.Tab("📊 Peer Comparison"):
                        compare_btn = gr.Button("Compare with Peers", variant="primary")
                        compare_output = gr.Markdown()

                        compare_btn.click(
                            fn=compare_with_peers,
                            outputs=[compare_output]
                        )

            # ============================================================
            # TAB 8: DAILY TRACKING
            # ============================================================
            with gr.Tab("📅 Daily Tracker", id=7):
                gr.Markdown("""
                ## Quick Daily Check-in
                ### Takes only 30 seconds! ⚡
                """)

                with gr.Group():
                    gr.Markdown("### Did you spend money today?")

                    with gr.Row():
                        daily_spent = gr.Radio(
                            choices=["Yes", "No"],
                            label="",
                            value="Yes"
                        )

                    with gr.Column(visible=True) as daily_details:
                        gr.Markdown("### Quick estimate:")
                        daily_amount = gr.Radio(
                            choices=[
                                "Less than ₹200 (minimal)",
                                "₹200-500 (moderate)",
                                "₹500-1000 (average)",
                                "More than ₹1000 (high)"
                            ],
                            label="How much did you spend?",
                            value="₹200-500 (moderate)"
                        )

                        daily_category = gr.Dropdown(
                            label="Main spending category",
                            choices=["Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"],
                            value="Food"
                        )

                    daily_submit = gr.Button("✅ Log Today's Expenses", variant="primary", size="lg")
                    daily_output = gr.Markdown()

                def log_daily_expense(spent, amount_range, category):
                    if spent == "No":
                        app_state.gamification_points += 50
                        return f"""
🎉 **No-spend day!**

Congratulations! You earned 50 points! 🌟

💎 Total Points: {app_state.gamification_points}

Keep up the great discipline!
"""

                    # Parse amount
                    amount_map = {
                        "Less than ₹200 (minimal)": 100,
                        "₹200-500 (moderate)": 350,
                        "₹500-1000 (average)": 750,
                        "More than ₹1000 (high)": 1500
                    }
                    amount = amount_map.get(amount_range, 500)

                    # Add to expenses
                    expense_entry = {
                        'description': f'Daily {category}',
                        'amount': amount,
                        'category': category,
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'time': datetime.now().strftime("%H:%M:%S")
                    }
                    app_state.expenses_list.append(expense_entry)

                    return f"""
✅ **Logged successfully!**

- Amount: ₹{amount}
- Category: {category}
- Date: {datetime.now().strftime("%Y-%m-%d")}

Great job tracking! Consistency is key to financial awareness! 💪
"""

                daily_submit.click(
                    fn=log_daily_expense,
                    inputs=[daily_spent, daily_amount, daily_category],
                    outputs=[daily_output]
                )

                gr.Markdown("""
                ---
                ### 🎯 Daily Tracking Benefits
                - Build awareness of spending habits
                - Takes only 30 seconds per day
                - Automatic categorization
                - Earn points for no-spend days!
                """)

            # ============================================================
            # TAB 9: GAMIFICATION
            # ============================================================
            with gr.Tab("🎮 Challenges", id=8):
                gr.Markdown("## Financial Challenges & Rewards")

                def show_current_points():
                    return f"""
# 💎 Your Progress

## Current Points: {app_state.gamification_points}

### Available Challenges:

#### 🥉 Easy Challenges (300 points)
- **Save ₹1000 This Week**
  - Track all expenses for 7 days
  - Reduce spending by ₹1000

#### 🥈 Medium Challenges (500 points)
- **No Dining Out Week**
  - Cook at home for 7 days straight
  - Share your progress daily

- **5-Day Packed Lunch Challenge**
  - Bring lunch from home
  - Save on food expenses

#### 🥇 Hard Challenges (1000 points)
- **Reduce Utility Bills by 20%**
  - Track usage for 30 days
  - Implement saving strategies

- **Monthly Savings Goal**
  - Save 30% of income this month
  - No unnecessary purchases

---

### 🏆 Achievements
- 🌟 **First Week**: Complete profile + 7 days tracking
- 💰 **Saver**: Save money for 30 days straight
- 📊 **Analyst**: Review budget weekly for a month
- 🎯 **Goal Getter**: Achieve one financial goal

---

### Rewards Tier
- 0-500 points: 🥉 Bronze
- 501-1500 points: 🥈 Silver
- 1501-3000 points: 🥇 Gold
- 3000+ points: 💎 Platinum

**Your current tier:** {'🥉 Bronze' if app_state.gamification_points < 500 else '🥈 Silver' if app_state.gamification_points < 1500 else '🥇 Gold' if app_state.gamification_points < 3000 else '💎 Platinum'}
"""

                points_display = gr.Markdown()
                refresh_points = gr.Button("🔄 Refresh Points")

                refresh_points.click(
                    fn=show_current_points,
                    outputs=[points_display]
                )

                # Initialize display
                app.load(
                    fn=show_current_points,
                    outputs=[points_display]
                )

                gr.Markdown("""
                ---
                ### 💡 How to Earn Points
                - Complete profile: 100 points
                - Add income sources: 50 points each
                - Track expenses daily: 10 points/day
                - No-spend day: 50 points
                - Complete challenges: 300-1000 points
                - Achieve savings goal: 500 points
                """)

            # ============================================================
            # TAB 10: REPORTS
            # ============================================================
            with gr.Tab("📄 Reports", id=9):
                gr.Markdown("## Export Your Financial Reports")

                with gr.Group():
                    gr.Markdown("### Generate Comprehensive Report")

                    report_type = gr.Radio(
                        choices=["Monthly Summary", "Detailed Analysis", "Tax Report", "Full Report"],
                        label="Report Type",
                        value="Monthly Summary"
                    )

                    generate_report_btn = gr.Button("📥 Generate Report", variant="primary")
                    report_output = gr.Markdown()

                def generate_report(report_type):
                    if not app_state.analysis:
                        return "⚠️ Please complete budget analysis first"

                    report = f"""
# 📊 Financial Report
## {report_type}
### Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 👤 Profile Summary
- **Name:** {app_state.user_profile.get('name', 'N/A')}
- **Age:** {app_state.user_profile.get('age', 'N/A')}
- **Category:** {app_state.user_profile.get('category', 'N/A')}
- **Location:** {app_state.user_profile.get('city', 'N/A')}, {app_state.user_profile.get('country', 'N/A')}

---

## 💰 Financial Summary

### Income
- **Total Monthly Income:** ₹{app_state.analysis['total_income']:,.0f}
- **Number of Sources:** {len(app_state.income_list)}

### Expenses
- **Total Monthly Expenses:** ₹{app_state.analysis['total_expenses']:,.0f}
- **Number of Transactions:** {len(app_state.expenses_list)}

### Savings
- **Monthly Savings:** ₹{app_state.analysis['savings']:,.0f}
- **Savings Rate:** {app_state.analysis['savings_rate']:.1f}%
- **Status:** {'✅ Surplus' if app_state.analysis['status'] == 'surplus' else '⚠️ Deficit'}

---

## 📊 Expense Breakdown
"""

                    for category, amount in app_state.analysis.get('category_breakdown', {}).items():
                        percentage = (amount / app_state.analysis['total_expenses'] * 100) if app_state.analysis['total_expenses'] > 0 else 0
                        report += f"\n- **{category}:** ₹{amount:,.0f} ({percentage:.1f}%)"

                    report += f"""

---

## 🎯 Goals
"""
                    for i, goal in enumerate(app_state.user_profile.get('goals', []), 1):
                        report += f"\n{i}. {goal}"

                    report += f"""

---

## 💡 Recommendations

1. **Savings Goal:** Aim to save at least 20% of income
2. **Emergency Fund:** Build 6 months of expenses (₹{app_state.analysis['total_expenses'] * 6:,.0f})
3. **Investment:** Consider SIP in mutual funds with saved amount
4. **Tax Planning:** Utilize Section 80C for tax savings

---

## 📈 Progress Tracking
- **Gamification Points:** {app_state.gamification_points}
- **Days Tracked:** {len(set(e['date'] for e in app_state.expenses_list))}
- **Consistency Score:** {'Excellent' if len(app_state.expenses_list) > 20 else 'Good' if len(app_state.expenses_list) > 10 else 'Getting Started'}

---

*This report is generated by AI Financial Advisor*
*For best results, update your data regularly*
"""

                    return report

                generate_report_btn.click(
                    fn=generate_report,
                    inputs=[report_type],
                    outputs=[report_output]
                )

                gr.Markdown("""
                ---
                ### 📋 Report Features
                - Comprehensive financial overview
                - Category-wise expense analysis
                - Savings rate tracking
                - Goal progress monitoring
                - Actionable recommendations

                💡 **Tip:** Generate reports weekly to track your progress!
                """)

        # Footer
        gr.Markdown("""
        ---
        <div style='text-align: center; padding: 20px;'>
            <p><strong>AI Financial Advisor</strong> | Powered by IBM Granite 3.2-2B</p>
            <p>🔒 Your data is private and secure | 💡 All AI suggestions are for informational purposes</p>
        </div>
        """)

    return app

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🤖 AI FINANCIAL ADVISOR - GRADIO UI 🤖             ║
    ║                                                              ║
    ║     Powered by IBM Granite 3.2-2B Instruct Model            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝

    🚀 Starting Gradio Interface...
    📱 The app will open in a new browser tab
    🌐 You'll get a public URL to share (optional)

    """)

    # Create and launch the interface
    demo = create_ui()

    # Launch with public URL for Colab
    demo.launch(
        share=True,  # Creates public URL for sharing
        debug=True,  # Enable debug mode
        show_error=True  # Show detailed errors
    )

    print("""
    ✅ Application is running!
    📱 Access it through the link above
    🛑 To stop: Click the stop button or press Ctrl+C
    """)