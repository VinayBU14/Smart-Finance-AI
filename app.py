"""
AI Financial Advisor - Optimized & Professional Version
Uses IBM Granite 3.2-2B Instruct Model
Optimized for speed and better responses
Run on Google Colab with Gradio UI

INSTALLATION:
!pip install -q transformers torch accelerate huggingface_hub pandas matplotlib seaborn plotly gradio
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.user_profile = {}
        self.income_list = []
        self.expenses_list = []
        self.analysis = {}
        self.is_model_loaded = False
        self.gamification_points = 0

app_state = AppState()

# ============================================================================
# OPTIMIZED MODEL LOADING WITH BETTER PROMPTING
# ============================================================================

def load_model():
    """Load IBM Granite model with optimizations"""
    if app_state.is_model_loaded:
        return "Model already loaded successfully."
    
    try:
        yield "Loading IBM Granite 3.2-2B model. This will take 2-3 minutes..."
        
        model_name = "ibm-granite/granite-3.2-2b-instruct"
        
        # Load tokenizer
        app_state.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True  # Faster tokenization
        )
        
        if app_state.tokenizer.pad_token is None:
            app_state.tokenizer.pad_token = app_state.tokenizer.eos_token
        
        yield "Tokenizer loaded. Loading model..."
        
        # Load model with optimizations
        app_state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_cache=True  # Enable KV cache for faster generation
        )
        
        app_state.model.eval()  # Set to evaluation mode
        
        app_state.is_model_loaded = True
        device = "GPU" if torch.cuda.is_available() else "CPU"
        yield f"Model loaded successfully on {device}. Ready to assist you."
        
    except Exception as e:
        yield f"Error loading model: {str(e)}\nPlease check your internet connection and try again."

# ============================================================================
# IMPROVED AI RESPONSE GENERATION
# ============================================================================

def generate_ai_response(prompt: str, context: str = "", max_tokens: int = 200) -> str:
    """Generate AI response with improved prompting"""
    if not app_state.is_model_loaded:
        return "Please load the AI model first from the Setup tab."
    
    try:
        # Construct better prompt for financial advisor
        system_prompt = """You are a professional financial advisor assistant. Provide clear, concise, and actionable financial advice. Be direct and helpful."""
        
        if context:
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {prompt}\n\nAdvice:"""
        else:
            full_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nAdvice:"""
        
        # Tokenize with proper settings
        inputs = app_state.tokenizer(
            full_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=800,  # Limit input length
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(app_state.model.device) for k, v in inputs.items()}
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = app_state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=30,  # Ensure substantial response
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_beams=1,  # Faster than beam search
                repetition_penalty=1.2,  # Avoid repetition
                pad_token_id=app_state.tokenizer.pad_token_id,
                eos_token_id=app_state.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode response
        full_response = app_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated advice (after "Advice:")
        if "Advice:" in full_response:
            response = full_response.split("Advice:")[-1].strip()
        else:
            response = full_response[len(full_prompt):].strip()
        
        # Clean up response
        response = response.replace("User Question:", "").strip()
        response = response.split("\n\n")[0]  # Take first paragraph for conciseness
        
        if not response or len(response) < 20:
            response = "I understand your question. Based on standard financial principles, I recommend: reviewing your budget carefully, identifying areas where you can reduce expenses, and setting clear savings goals. Would you like specific advice on any particular aspect of your finances?"
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try rephrasing your question."

# ============================================================================
# USER PROFILE MANAGEMENT
# ============================================================================

def save_basic_info(name, age, category, marital_status):
    """Save basic user information"""
    if not name or not age:
        return "Please fill in all required fields.", None
    
    app_state.user_profile.update({
        'name': name,
        'age': int(age),
        'category': category,
        'marital_status': marital_status,
        'created_at': datetime.now().strftime("%Y-%m-%d")
    })
    
    summary = f"""Profile Created Successfully

Name: {name}
Age: {age}
Category: {category}
Marital Status: {marital_status}

Please continue to complete your profile."""
    
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
    
    return f"Family and location information saved. Total family members: {family_size}, Location: {city}, {country}"

def save_assets(owns_vehicle, vehicle_type, owns_property):
    """Save assets information"""
    app_state.user_profile.update({
        'owns_vehicle': owns_vehicle,
        'vehicle_type': vehicle_type if owns_vehicle else 'None',
        'owns_property': owns_property
    })
    
    return "Asset information saved successfully."

def save_goals(goals_text):
    """Save financial goals"""
    goals = [g.strip() for g in goals_text.split('\n') if g.strip()]
    app_state.user_profile['goals'] = goals
    
    profile_summary = f"""Profile Setup Complete

Summary:
- Name: {app_state.user_profile.get('name')}
- Category: {app_state.user_profile.get('category')}
- Family Size: {app_state.user_profile.get('family_size')}
- Location: {app_state.user_profile.get('city')}
- Financial Goals: {len(goals)} goals set

Next Step: Add your income sources in the Income tab."""
    
    return profile_summary

# ============================================================================
# INCOME MANAGEMENT
# ============================================================================

def add_income(source, amount, frequency):
    """Add income entry"""
    if not source or not amount:
        return "Please fill in all fields.", create_income_table()
    
    try:
        income_entry = {
            'source': source,
            'amount': float(amount),
            'frequency': frequency,
            'date': datetime.now().strftime("%Y-%m-%d")
        }
        
        app_state.income_list.append(income_entry)
        
        total = sum(inc['amount'] for inc in app_state.income_list)
        return f"Income added: Rs.{amount:,.0f} from {source}. Total monthly income: Rs.{total:,.0f}", create_income_table()
        
    except ValueError:
        return "Invalid amount entered.", create_income_table()

def create_income_table():
    """Create income summary table"""
    if not app_state.income_list:
        return pd.DataFrame(columns=['Source', 'Amount', 'Frequency'])
    
    df = pd.DataFrame(app_state.income_list)
    df_display = pd.DataFrame({
        'Source': df['source'],
        'Amount': df['amount'].apply(lambda x: f"Rs.{x:,.0f}"),
        'Frequency': df['frequency']
    })
    
    return df_display

def get_income_sources(category):
    """Get income sources based on category"""
    sources = {
        'Student': ['Pocket Money', 'Scholarship', 'Part-time Job', 'Tuition', 'Freelancing'],
        'Employed': ['Salary', 'Bonus', 'Investment Returns', 'Rental Income', 'Side Business'],
        'Unemployed': ['Savings', 'Family Support', 'Freelancing', 'Benefits'],
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
        return "Please enter an expense.", create_expense_table(), None
    
    try:
        parts = description.strip().split()
        amount = float(parts[-1])
        desc = ' '.join(parts[:-1]) if len(parts) > 1 else 'Expense'
        
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
            f"Expense added: Rs.{amount:,.0f} - {desc} (Category: {category})\nTotal expenses: Rs.{total:,.0f}",
            create_expense_table(),
            ""
        )
        
    except (ValueError, IndexError):
        return "Invalid format. Use: 'description amount' (e.g., 'Groceries 500')", create_expense_table(), description

def auto_categorize_expense(description: str) -> str:
    """Auto-categorize based on keywords"""
    description = description.lower()
    
    keywords = {
        'Food & Groceries': ['grocery', 'food', 'vegetables', 'supermarket', 'lunch', 'dinner', 'breakfast', 'restaurant'],
        'Rent': ['rent', 'lease', 'housing'],
        'Utilities': ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile', 'wifi'],
        'Transportation': ['petrol', 'diesel', 'fuel', 'bus', 'taxi', 'uber', 'ola', 'auto', 'transport'],
        'Healthcare': ['medicine', 'doctor', 'hospital', 'health', 'medical', 'pharmacy'],
        'Education': ['school', 'college', 'tuition', 'books', 'course', 'fees', 'education'],
        'Entertainment': ['movie', 'restaurant', 'party', 'gaming', 'netflix', 'prime', 'entertainment'],
        'Clothing': ['clothes', 'shoes', 'dress', 'shirt', 'fashion', 'clothing']
    }
    
    for category, words in keywords.items():
        if any(word in description for word in words):
            return category
    
    return 'Other'

def add_expense_detailed(description, amount, category):
    """Detailed expense entry"""
    if not description or not amount:
        return "Please fill in all fields.", create_expense_table()
    
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
        return f"Expense added: Rs.{amount:,.0f} - {description}. Total: Rs.{total:,.0f}", create_expense_table()
        
    except ValueError:
        return "Invalid amount entered.", create_expense_table()

def create_expense_table():
    """Create expense summary table"""
    if not app_state.expenses_list:
        return pd.DataFrame(columns=['Date', 'Description', 'Category', 'Amount'])
    
    df = pd.DataFrame(app_state.expenses_list)
    df_display = pd.DataFrame({
        'Date': df['date'],
        'Description': df['description'],
        'Category': df['category'],
        'Amount': df['amount'].apply(lambda x: f"Rs.{x:,.0f}")
    })
    
    return df_display.tail(10)

# ============================================================================
# BUDGET ANALYSIS WITH AI
# ============================================================================

def analyze_budget():
    """Analyze budget with AI insights"""
    if not app_state.income_list or not app_state.expenses_list:
        return "Please add income and expenses before analyzing.", None, None, None
    
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
    summary = f"""Financial Analysis Report

OVERVIEW
Total Monthly Income: Rs.{total_income:,.0f}
Total Monthly Expenses: Rs.{total_expenses:,.0f}
Net Savings: Rs.{savings:,.0f}
Savings Rate: {savings_rate:.1f}%

STATUS: {'Budget Surplus' if savings > 0 else 'Budget Deficit'}
"""
    
    if savings < 0:
        summary += f"\nYou are overspending by Rs.{abs(savings):,.0f} per month.\n"
    
    # Top expense categories
    if category_totals:
        sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        summary += "\nTOP EXPENSE CATEGORIES:\n"
        for cat, amt in sorted_cats[:3]:
            pct = (amt / total_expenses * 100) if total_expenses > 0 else 0
            summary += f"- {cat}: Rs.{amt:,.0f} ({pct:.1f}%)\n"
    
    # Get AI insights
    if app_state.is_model_loaded:
        summary += "\n--- AI FINANCIAL ADVISOR INSIGHTS ---\n\n"
        
        context = f"""User Financial Data:
- Category: {app_state.user_profile.get('category', 'Unknown')}
- Age: {app_state.user_profile.get('age', 'Unknown')}
- Monthly Income: Rs.{total_income:,.0f}
- Monthly Expenses: Rs.{total_expenses:,.0f}
- Savings: Rs.{savings:,.0f}
- Savings Rate: {savings_rate:.1f}%
- Top Expense: {sorted_cats[0][0] if category_totals else 'None'} (Rs.{sorted_cats[0][1]:,.0f} if category_totals else 0)
"""
        
        query = "Based on this financial data, provide 3 specific actionable recommendations to improve the financial situation."
        ai_insights = generate_ai_response(query, context, max_tokens=250)
        summary += ai_insights
    else:
        summary += "\nLoad the AI model to get personalized insights."
    
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
        go.Bar(
            x=['Income', 'Expenses', 'Savings'],
            y=[
                app_state.analysis['total_income'],
                app_state.analysis['total_expenses'],
                app_state.analysis['savings']
            ],
            marker_color=['#27ae60', '#e74c3c', '#3498db'],
            text=[
                f"Rs.{app_state.analysis['total_income']:,.0f}",
                f"Rs.{app_state.analysis['total_expenses']:,.0f}",
                f"Rs.{app_state.analysis['savings']:,.0f}"
            ],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Income vs Expenses vs Savings',
        yaxis_title='Amount (Rs.)',
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_expense_breakdown_chart():
    """Create expense breakdown pie chart"""
    if not app_state.analysis or not app_state.analysis.get('category_breakdown'):
        return None
    
    categories = list(app_state.analysis['category_breakdown'].keys())
    amounts = list(app_state.analysis['category_breakdown'].values())
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=amounts,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Rs.%{value:,.0f}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Expense Distribution by Category',
        height=400,
        template='plotly_white'
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
        delta={'reference': 20, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 10], 'color': "#ffcccc"},
                {'range': [10, 20], 'color': "#ffffcc"},
                {'range': [20, 50], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig.update_layout(height=400, template='plotly_white')
    
    return fig

# ============================================================================
# CHATBOT WITH IMPROVED RESPONSES
# ============================================================================

def chatbot_response(message, history):
    """Professional chatbot with context"""
    if not app_state.is_model_loaded:
        return "Please load the AI model first from the Setup tab."
    
    if not message.strip():
        return "Please enter a question."
    
    # Build context from user profile and financial data
    context_parts = []
    
    if app_state.user_profile:
        context_parts.append(f"User Profile: {app_state.user_profile.get('category', 'Unknown')}, Age {app_state.user_profile.get('age', 'Unknown')}")
    
    if app_state.analysis:
        context_parts.append(f"Financial Status: Monthly Income Rs.{app_state.analysis['total_income']:,.0f}, Expenses Rs.{app_state.analysis['total_expenses']:,.0f}, Savings Rs.{app_state.analysis['savings']:,.0f}")
    
    context = "\n".join(context_parts) if context_parts else ""
    
    response = generate_ai_response(message, context, max_tokens=200)
    
    return response

# ============================================================================
# ADDITIONAL FEATURES
# ============================================================================

def get_tax_suggestions():
    """Tax saving suggestions"""
    if not app_state.analysis:
        return "Please complete budget analysis first."
    
    annual_income = app_state.analysis['total_income'] * 12
    
    return f"""Tax Saving Opportunities (India)

Based on your annual income of Rs.{annual_income:,.0f}:

SECTION 80C (Maximum Deduction: Rs.1,50,000)
Save up to Rs.46,800 in taxes

Investment Options:
- Public Provident Fund (PPF): 7.1% interest, 15-year lock-in
- Equity Linked Savings Scheme (ELSS): Market-linked returns, 3-year lock-in
- Life Insurance Premium: Protection + tax benefit
- National Savings Certificate (NSC): Fixed returns, 5-year tenure
- Employee Provident Fund (EPF): Mandatory for salaried employees

SECTION 80D (Health Insurance)
Deduction: Rs.25,000 (Rs.50,000 for senior citizens)
- Health insurance premium for self and family
- Preventive health check-up: Rs.5,000

SECTION 80CCD(1B) (National Pension Scheme)
Additional deduction: Rs.50,000
- Long-term retirement planning
- Tax benefits + market returns

ESTIMATED TAX SAVINGS
Maximum potential annual savings: Rs.70,200

Recommendation: Start with PPF (Rs.12,500/month) and health insurance for immediate tax benefits."""

def get_government_schemes():
    """Government schemes information"""
    if not app_state.user_profile:
        return "Please complete your profile first."
    
    category = app_state.user_profile.get('category', '').lower()
    age = app_state.user_profile.get('age', 0)
    
    schemes = "Applicable Government Schemes\n\n"
    
    if 'student' in category:
        schemes += """FOR STUDENTS:

1. PM Scholarship Scheme
   Benefit: Rs.2,000-3,000 per month
   Eligibility: Merit-based for economically weaker sections
   Apply: scholarships.gov.in

2. National Scholarship Portal (NSP)
   Benefit: Various state and central scholarships
   Eligibility: Based on category and merit

3. Post Matric Scholarship
   Benefit: Full tuition + maintenance allowance
   Eligibility: SC/ST/OBC students
"""
    
    if 'employed' in category:
        schemes += """FOR EMPLOYED:

1. PM Suraksha Bima Yojana
   Benefit: Rs.2 lakh accident insurance
   Premium: Rs.20 per year
   Eligibility: Age 18-70 with bank account

2. Atal Pension Yojana
   Benefit: Guaranteed pension Rs.1,000-5,000 per month
   Eligibility: Age 18-40
"""
    
    if 'housewife' in category:
        schemes += """FOR HOMEMAKERS:

1. Mahila Samman Savings Certificate
   Benefit: 7.5% interest (higher than FD)
   Eligibility: All women
   Tenure: 2 years

2. PM Jan Dhan Yojana
   Benefit: Free bank account + Rs.2 lakh insurance
   Eligibility: All Indian citizens
"""
    
    if age >= 60:
        schemes += """FOR SENIOR CITIZENS:

1. Senior Citizen Savings Scheme (SCSS)
   Benefit: 8.2% interest per annum
   Maximum: Rs.30 lakh investment
   Eligibility: Age 60+

2. Pradhan Mantri Vaya Vandana Yojana
   Benefit: Guaranteed pension
   Returns: 7.4% per annum
"""
    
    schemes += """\nUNIVERSAL SCHEMES:

PM Suraksha Bima Yojana: Rs.2 lakh accident cover for Rs.20/year
PM Jeevan Jyoti Bima Yojana: Rs.2 lakh life cover for Rs.436/year

Visit your nearest bank or post office to apply for these schemes."""
    
    return schemes

def compare_with_peers():
    """Compare with peer benchmarks"""
    if not app_state.analysis:
        return "Please complete budget analysis first."
    
    category = app_state.user_profile.get('category', 'employed').lower()
    
    benchmarks = {
        'student': {'avg_savings_rate': 18.75},
        'employed': {'avg_savings_rate': 22.22},
        'housewife': {'avg_savings_rate': 15.0},
        'retired': {'avg_savings_rate': 26.67}
    }
    
    benchmark = benchmarks.get(category, benchmarks['employed'])
    user_savings_rate = app_state.analysis['savings_rate']
    savings_diff = user_savings_rate - benchmark['avg_savings_rate']
    
    comparison = f"""Peer Comparison Report

Your Category: {category.title()}

SAVINGS RATE COMPARISON
Your Savings Rate: {user_savings_rate:.1f}%
Peer Average: {benchmark['avg_savings_rate']:.1f}%
Difference: {savings_diff:+.1f}%

"""
    
    if savings_diff > 5:
        comparison += f"""ASSESSMENT: Excellent Performance
You are saving {savings_diff:.1f}% more than your peers. You are in the top 20% of savers in your category. Continue your disciplined approach to achieve your financial goals."""
    elif savings_diff > 0:
        comparison += f"""ASSESSMENT: Above Average
You are performing better than the average in your category. Small improvements can elevate you to the top tier of savers."""
    else:
        comparison += f"""ASSESSMENT: Opportunity for Improvement
You are saving {abs(savings_diff):.1f}% less than peers in your category.

Action Plan:
1. Review your expense categories and identify non-essential spending
2. Set a specific savings goal for next month
3. Aim to increase savings rate by 5% within 3 months
4. Track expenses daily to maintain awareness"""
    
    return comparison

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_ui():
    """Create professional Gradio interface"""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    )
    
    with gr.Blocks(title="AI Financial Advisor", theme=theme) as app:
        
        gr.Markdown("""
        # AI Financial Advisor
        ### Professional Financial Management powered by IBM Granite 3.2-2B
        """)
        
        with gr.Tabs() as tabs:
            
            # TAB 1: SETUP
            with gr.Tab("Setup"):
                gr.Markdown("""
                ## Getting Started
                
                **Step 1:** Load the AI model (2-3 minutes first time)
                **Step 2:** Complete your profile in the Profile tab
                **Step 3:** Add your income and expenses
                **Step 4:** Get AI-powered financial insights
                """)
                
                load_btn = gr.Button("Load AI Model", variant="primary", size="lg")
                model_status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                load_btn.click(fn=load_model, outputs=[model_status])
                
            # TAB 2: PROFILE
            with gr.Tab("Profile"):
                gr.Markdown("## User Profile")
                
                with gr.Group():
                    gr.Markdown("### Basic Information")
                    with gr.Row():
                        name_input = gr.Textbox(label="Full Name", placeholder="Enter your name")
                        age_input = gr.Number(label="Age", value=25, minimum=18, maximum=100)
                    
                    with gr.Row():
                        category_input = gr.Dropdown(
                            label="Occupation Category",
                            choices=["Student", "Employed", "Unemployed", "Housewife", "Retired"],
                            value="Employed"
                        )
                        marital_input = gr.Dropdown(
                            label="Marital Status",
                            choices=["Single", "Married", "Divorced", "Widowed"],
                            value="Single"
                        )
                    
                    basic_submit = gr.Button("Save Basic Information", variant="primary")
                    basic_output = gr.Textbox(label="Status", lines=6, interactive=False)
                
                with gr.Group(visible=True):
                    gr.Markdown("### Family & Location")
                    with gr.Row():
                        family_size_input = gr.Number(label="Total Family Members", value=1, minimum=1)
                        dependents_input = gr.Number(label="Dependents", value=0, minimum=0)
                    
                    with gr.Row():
                        city_input = gr.Textbox(label="City", placeholder="e.g., Mumbai, Bangalore")
                        country_input = gr.Textbox(label="Country", value="India")
                    
                    living_input = gr.Dropdown(
                        label="Living Situation",
                        choices=["Own Home", "Rented", "PG", "Hostel", "With Parents"],
                        value="Rented"
                    )
                    
                    family_submit = gr.Button("Save Family Information")
                    family_output = gr.Textbox(label="Status", lines=2, interactive=False)
                
                with gr.Group():
                    gr.Markdown("### Assets")
                    owns_vehicle = gr.Checkbox(label="Own Vehicle")
                    vehicle_type = gr.Dropdown(
                        label="Vehicle Type",
                        choices=["Car", "Bike", "Both", "None"],
                        value="None"
                    )
                    owns_property = gr.Checkbox(label="Own Property")
                    
                    assets_submit = gr.Button("Save Asset Information")
                    assets_output = gr.Textbox(label="Status", lines=2, interactive=False)
                
                with gr.Group():
                    gr.Markdown("### Financial Goals")
                    goals_input = gr.Textbox(
                        label="List your financial goals (one per line)",
                        placeholder="Example:\nBuy a car within 2 years\nSave for children's education\nPlan international trip",
                        lines=5
                    )
                    
                    goals_submit = gr.Button("Complete Profile", variant="primary")
                    profile_summary = gr.Textbox(label="Profile Summary", lines=8, interactive=False)
                
                # Connect buttons
                basic_submit.click(
                    fn=save_basic_info,
                    inputs=[name_input, age_input, category_input, marital_input],
                    outputs=[basic_output, gr.State()]
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
            
            # TAB 3: INCOME
            with gr.Tab("Income"):
                gr.Markdown("## Income Management")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        income_source = gr.Dropdown(
                            label="Income Source",
                            choices=["Salary", "Bonus", "Investment Returns", "Rental Income", "Side Business", "Freelancing"],
                            value="Salary"
                        )
                        income_amount = gr.Number(label="Monthly Amount (Rs.)", value=0, minimum=0)
                        income_frequency = gr.Dropdown(
                            label="Frequency",
                            choices=["Monthly", "Yearly", "One-time"],
                            value="Monthly"
                        )
                        
                        add_income_btn = gr.Button("Add Income", variant="primary")
                        income_output = gr.Textbox(label="Status", lines=2, interactive=False)
                    
                    with gr.Column(scale=3):
                        income_table = gr.Dataframe(
                            label="Your Income Sources",
                            headers=["Source", "Amount", "Frequency"],
                            interactive=False
                        )
                
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
            
            # TAB 4: EXPENSES
            with gr.Tab("Expenses"):
                gr.Markdown("## Expense Tracking")
                
                with gr.Tabs():
                    with gr.Tab("Quick Entry"):
                        gr.Markdown("""
                        ### Quick Expense Entry
                        Enter expenses naturally. Format: description amount
                        
                        Examples:
                        - Groceries 2500
                        - Lunch 250
                        - Electricity bill 1500
                        """)
                        
                        quick_expense_input = gr.Textbox(
                            label="Quick Entry",
                            placeholder="e.g., Lunch 250",
                            lines=1
                        )
                        quick_add_btn = gr.Button("Add Expense", variant="primary")
                        quick_expense_output = gr.Textbox(label="Status", lines=2, interactive=False)
                    
                    with gr.Tab("Detailed Entry"):
                        with gr.Row():
                            expense_desc = gr.Textbox(label="Description", placeholder="e.g., Monthly groceries")
                            expense_amount = gr.Number(label="Amount (Rs.)", value=0, minimum=0)
                        
                        expense_category = gr.Dropdown(
                            label="Category",
                            choices=[
                                "Food & Groceries", "Rent", "Utilities", "Transportation",
                                "Healthcare", "Education", "Entertainment", "Clothing",
                                "Insurance", "EMI/Loans", "Communication", "Personal Care",
                                "Household Items", "Gifts & Donations", "Travel", "Other"
                            ],
                            value="Food & Groceries"
                        )
                        
                        detailed_add_btn = gr.Button("Add Expense", variant="primary")
                        detailed_expense_output = gr.Textbox(label="Status", lines=2, interactive=False)
                
                gr.Markdown("### Recent Expenses")
                expense_table = gr.Dataframe(
                    label="Last 10 Expenses",
                    headers=["Date", "Description", "Category", "Amount"],
                    interactive=False
                )
                
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
            
            # TAB 5: ANALYSIS
            with gr.Tab("Analysis"):
                gr.Markdown("## Financial Analysis")
                
                analyze_btn = gr.Button("Analyze My Finances", variant="primary", size="lg")
                
                analysis_summary = gr.Textbox(label="Analysis Report", lines=20, interactive=False)
                
                with gr.Row():
                    chart1 = gr.Plot(label="Income vs Expenses")
                    chart2 = gr.Plot(label="Expense Breakdown")
                
                chart3 = gr.Plot(label="Savings Rate Gauge")
                
                analyze_btn.click(
                    fn=analyze_budget,
                    outputs=[analysis_summary, chart1, chart2, chart3]
                )
            
            # TAB 6: AI ASSISTANT
            with gr.Tab("AI Assistant"):
                gr.Markdown("""
                ## Financial Advisory Chat
                
                Ask questions about your finances and get personalized advice from the AI.
                
                Example questions:
                - How can I improve my savings rate?
                - What expenses should I prioritize cutting?
                - Should I invest in mutual funds or fixed deposits?
                - How much emergency fund should I maintain?
                """)
                
                chatbot = gr.Chatbot(height=500, label="AI Financial Advisor")
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your financial question here...",
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
            
            # TAB 7: RECOMMENDATIONS
            with gr.Tab("Recommendations"):
                gr.Markdown("## Expert Financial Recommendations")
                
                with gr.Tabs():
                    with gr.Tab("Tax Savings"):
                        tax_btn = gr.Button("Get Tax Saving Suggestions", variant="primary")
                        tax_output = gr.Textbox(label="Tax Saving Opportunities", lines=25, interactive=False)
                        
                        tax_btn.click(fn=get_tax_suggestions, outputs=[tax_output])
                    
                    with gr.Tab("Government Schemes"):
                        schemes_btn = gr.Button("Show Applicable Schemes", variant="primary")
                        schemes_output = gr.Textbox(label="Government Schemes", lines=25, interactive=False)
                        
                        schemes_btn.click(fn=get_government_schemes, outputs=[schemes_output])
                    
                    with gr.Tab("Peer Comparison"):
                        compare_btn = gr.Button("Compare with Peers", variant="primary")
                        compare_output = gr.Textbox(label="Peer Comparison", lines=20, interactive=False)
                        
                        compare_btn.click(fn=compare_with_peers, outputs=[compare_output])
            
            # TAB 8: REPORTS
            with gr.Tab("Reports"):
                gr.Markdown("## Financial Reports")
                
                def generate_report():
                    if not app_state.analysis:
                        return "Please complete budget analysis first."
                    
                    report = f"""FINANCIAL REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PROFILE SUMMARY
Name: {app_state.user_profile.get('name', 'N/A')}
Category: {app_state.user_profile.get('category', 'N/A')}
Location: {app_state.user_profile.get('city', 'N/A')}

FINANCIAL SUMMARY
Monthly Income: Rs.{app_state.analysis['total_income']:,.0f}
Monthly Expenses: Rs.{app_state.analysis['total_expenses']:,.0f}
Monthly Savings: Rs.{app_state.analysis['savings']:,.0f}
Savings Rate: {app_state.analysis['savings_rate']:.1f}%

EXPENSE BREAKDOWN
"""
                    
                    for category, amount in app_state.analysis.get('category_breakdown', {}).items():
                        percentage = (amount / app_state.analysis['total_expenses'] * 100) if app_state.analysis['total_expenses'] > 0 else 0
                        report += f"{category}: Rs.{amount:,.0f} ({percentage:.1f}%)\n"
                    
                    report += f"""

FINANCIAL GOALS
"""
                    for i, goal in enumerate(app_state.user_profile.get('goals', []), 1):
                        report += f"{i}. {goal}\n"
                    
                    report += f"""

RECOMMENDATIONS
1. Maintain emergency fund of 6 months expenses: Rs.{app_state.analysis['total_expenses'] * 6:,.0f}
2. Target savings rate: 20% minimum
3. Review and optimize expenses monthly
4. Consider tax-saving investments under Section 80C

ANALYSIS DATE: {datetime.now().strftime("%Y-%m-%d")}
"""
                    
                    return report
                
                generate_report_btn = gr.Button("Generate Comprehensive Report", variant="primary", size="lg")
                report_output = gr.Textbox(label="Financial Report", lines=30, interactive=False)
                
                generate_report_btn.click(fn=generate_report, outputs=[report_output])
        
        gr.Markdown("""
        ---
        **AI Financial Advisor** | Powered by IBM Granite 3.2-2B | Professional Financial Management Tool
        
        *Disclaimer: This tool provides general financial guidance for informational purposes only. 
        Consult a qualified financial advisor for personalized financial planning.*
        """)
    
    return app

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       AI FINANCIAL ADVISOR - OPTIMIZED VERSION              ║
    ║       Powered by IBM Granite 3.2-2B                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Starting application...
    """)
    
    demo = create_ui()
    
    # For Colab: use share=True to get public URL
    # For deployment: remove share parameter
    demo.launch(
        share=True,
        debug=False,
        show_error=True
    )
    
    print("Application is running!")
    print("Access the interface through the link above.")
