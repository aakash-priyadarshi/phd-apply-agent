# 🎓 PhD Outreach Automation Agent

A sophisticated, AI-powered application that automates the process of discovering professors, generating personalized outreach emails, and managing PhD application communications. Built with Streamlit, OpenAI GPT models, and Gmail API integration.

## 🚀 Features

### 🔍 **Intelligent Professor Discovery**
- **Stage 1**: Cost-effective professor discovery using GPT-4o-mini
- Advanced web scraping with BeautifulSoup and requests
- Smart alignment scoring based on research interests
- Duplicate detection and management
- University-specific faculty page navigation

### 📧 **AI-Powered Email Generation**
- **Stage 2**: High-quality personalized emails using GPT-4
- Contextual email drafting based on professor's research
- Professional email templates with customization
- Bulk email generation with cost optimization
- Email preview and editing with modal interface

### 📄 **CV Analysis & Profile Generation**
- Automated CV parsing using PyPDF2
- AI-generated research profiles from CV content
- Persistent profile storage across app restarts
- Research alignment highlighting

### 📬 **Gmail Integration**
- Seamless Gmail API integration for email sending
- Bulk email sending with rate limiting
- Email delivery tracking and status monitoring
- CV attachment support
- Professional email formatting

### 💰 **Cost Tracking & Analytics**
- Real-time API cost monitoring
- Stage-wise cost breakdown (Discovery vs Email Generation)
- Daily and total cost summaries
- Professor processing statistics

### 🎨 **Modern User Interface**
- Clean, intuitive Streamlit interface
- Responsive design with custom CSS styling
- Modal dialogs for spacious email editing
- Progress tracking and live updates
- Professional dark/light theme support

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Google Cloud Console project with Gmail API enabled
- Gmail API credentials

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/aakash-priyadarshi/phd-apply-agent.git
cd phd-apply-agent
```

2. **Create virtual environment**
```bash
python -m venv phd_outreach_env
# On Windows
phd_outreach_env\Scripts\activate
# On macOS/Linux
source phd_outreach_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment setup**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "USER_NAME=Your Full Name" >> .env
echo "USER_EMAIL=your.email@example.com" >> .env
```

5. **Gmail API setup**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials as `credentials.json`
   - Place in project root directory

6. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 📚 Usage Guide

### 🔧 **Initial Configuration**

1. **API Setup**
   - Enter your OpenAI API key in the sidebar
   - Configure your name and email in User Settings
   - Verify Gmail connection status

2. **CV Upload & Analysis**
   - Upload your CV (PDF format)
   - Click "Analyze CV" to generate research profile
   - Review and customize the generated profile

### 🎯 **Professor Discovery (Stage 1)**

1. **Target Universities**
   - Add universities using the form
   - Specify departments and priority levels
   - Configure research focus areas

2. **Run Stage 1**
   - Click "🔍 Run Stage 1" for cost-effective discovery
   - Monitor progress in real-time
   - Review discovered professors with alignment scores

### ✉️ **Email Generation (Stage 2)**

1. **Individual Emails**
   - Select verified professors
   - Click "📧 Generate Email" for personalized content
   - Preview and edit emails in modal interface
   - Send individual emails with one click

2. **Bulk Operations**
   - Generate emails for all verified professors
   - Send all drafted emails with customizable delays
   - Complete generate & send automation

### 📊 **Monitoring & Management**

- **Cost Tracking**: Monitor API usage and costs
- **Database Management**: Clean duplicate entries
- **Progress Analytics**: Track email generation and sending
- **Status Management**: Monitor professor verification states

## 🏗️ Architecture

### **Two-Stage System**
```
Stage 1 (Discovery) → GPT-4o-mini → Cost-effective professor finding
Stage 2 (Emails)   → GPT-4       → High-quality personalized emails
```

### **Core Components**
- **`streamlit_app.py`**: Main application interface and orchestration
- **`gmail_manager.py`**: Gmail API integration and email handling
- **Database**: SQLite for professor data and cost tracking
- **API Manager**: OpenAI integration with cost monitoring
- **Web Scraper**: University faculty page processing

### **Data Flow**
1. **CV Analysis** → Research Profile Generation
2. **University Targeting** → Faculty Page Scraping
3. **Professor Discovery** → Alignment Scoring
4. **Email Generation** → Personalized Content Creation
5. **Email Sending** → Gmail API Delivery

## 📁 Project Structure

```
phd-apply-agent/
├── streamlit_app.py           # Main application
├── gmail_manager.py           # Gmail API integration
├── requirements.txt           # Python dependencies
├── credentials.json           # Gmail API credentials (not in repo)
├── .env                      # Environment variables (not in repo)
├── phd_outreach.db           # SQLite database
├── uploaded_cv.pdf           # User's CV (generated)
├── research_profile.txt      # Generated research profile
├── test_db_structure.py      # Database testing utility
├── test_persistence.py       # CV persistence testing
└── README.md                 # This file
```

## 🔧 Configuration

### **Environment Variables**
```env
OPENAI_API_KEY=your_openai_api_key
USER_NAME=Your Full Name
USER_EMAIL=your.email@example.com
```

### **Gmail API Setup**
1. Enable Gmail API in Google Cloud Console
2. Create OAuth 2.0 credentials
3. Download as `credentials.json`
4. First run will require browser authentication

### **OpenAI Models**
- **Stage 1**: `gpt-4o-mini` (cost-efficient discovery)
- **Stage 2**: `gpt-4` (high-quality email generation)
- **CV Analysis**: `gpt-4o-mini` (profile generation)

## 💡 Tips & Best Practices

### **Cost Optimization**
- Use Stage 1 for bulk professor discovery
- Generate emails selectively for high-alignment professors
- Monitor costs in real-time via dashboard

### **Email Quality**
- Ensure comprehensive CV upload for better personalization
- Review and customize generated emails before sending
- Use appropriate sending delays to avoid rate limiting

### **Data Management**
- Regularly clean duplicate professors
- Backup your database and research profile
- Monitor email delivery status

## 🚨 Troubleshooting

### **Common Issues**

**Gmail Authentication Errors**
```bash
# Re-authenticate Gmail
rm gmail_token.pickle
# Restart app and complete OAuth flow
```

**Database Errors**
```bash
# Run database structure test
python test_db_structure.py
```

**CV Persistence Issues**
```bash
# Test CV and profile persistence
python test_persistence.py
```

**JSON Parsing Errors**
- Fixed in latest version with control character cleaning
- Automatic fallback email extraction implemented

## 🔒 Privacy & Security

- **Local Processing**: All data stored locally in SQLite
- **API Security**: Credentials stored securely with proper scoping
- **Email Privacy**: Direct Gmail API integration, no third-party storage
- **Research Data**: CV and profile data remains on your machine

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, please:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include relevant logs and error messages

## 🚀 Roadmap

- [ ] Multi-language email templates
- [ ] Advanced professor filtering
- [ ] Email template customization
- [ ] Integration with academic databases
- [ ] Mobile-responsive interface
- [ ] Advanced analytics dashboard

## 🙏 Acknowledgments

- OpenAI for GPT models
- Google for Gmail API
- Streamlit for the web framework
- BeautifulSoup for web scraping capabilities

---

**Made with ❤️ for aspiring PhD students worldwide**

*Streamline your PhD applications with AI-powered automation*