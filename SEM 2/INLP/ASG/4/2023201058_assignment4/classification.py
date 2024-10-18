from ELMofunc import *
from ELMo import vocab, elmo_embeddings, train_loader, valid_loader, test_loader
# Instantiate the AG_News Analysis model
ag_news_model = AG_News_Analysis(len(vocab), elmo_embeddings).to(device)
print(ag_news_model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()


# Instantiate the AG_News Analysis model
ag_news_model_flow = AG_News_Analysis_Flow(len(vocab), elmo_embeddings).to(device)
print(ag_news_model_flow)

# Define the loss function and optimizer
criterion_flow = nn.CrossEntropyLoss()


# Instantiate the AG_News Analysis model
ag_news_model_froz = AG_News_Analysis_Froz(len(vocab), elmo_embeddings).to(device)
print(ag_news_model_froz)

# Define the loss function and optimizer
criterion_froz = nn.CrossEntropyLoss()


# Instantiate the AG_News Analysis model
ag_news_model_LNNF = AG_News_Analysis_LNNF(len(vocab), elmo_embeddings).to(device)
print(ag_news_model_LNNF)

# Define the loss function and optimizer
criterion_LNNF = nn.CrossEntropyLoss()


num_epochs = 16
BVA21 = 0.0
BTA21 = 0.0
BVA22 = 0.0
BTA22 = 0.0
BVA23 = 0.0
BTA23 = 0.0
BVA24 = 0.0
BTA24 = 0.0

optimizer = torch.optim.Adam(ag_news_model.parameters(), lr=1e-2)
ag_news_model, BVA21, BTA21 = DownStreamTask(ag_news_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model', BVA21, BTA21)

optimizer = torch.optim.Adam(ag_news_model.parameters(), lr=1e-3)
ag_news_model, BVA21, BTA21 = DownStreamTask(ag_news_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model', BVA21, BTA21)

optimizer = torch.optim.Adam(ag_news_model.parameters(), lr=1e-4)
ag_news_model, BVA21, BTA21 = DownStreamTask(ag_news_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model', BVA21, BTA21)

optimizer = torch.optim.Adam(ag_news_model.parameters(), lr=1e-5)
ag_news_model, BVA21, BTA21 = DownStreamTask(ag_news_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model', BVA21, BTA21)

optimizer = torch.optim.Adam(ag_news_model.parameters(), lr=1e-6)
ag_news_model, BVA21, BTA21 = DownStreamTask(ag_news_model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model', BVA21, BTA21)



optimizer_flow = torch.optim.Adam(ag_news_model_flow.parameters(), lr=1e-2)
ag_news_model_flow, BVA22, BTA22 = DownStreamTask(ag_news_model_flow, criterion_flow, optimizer_flow, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_flow', BVA22, BTA22)

optimizer_flow = torch.optim.Adam(ag_news_model_flow.parameters(), lr=1e-3)
ag_news_model_flow, BVA22, BTA22 = DownStreamTask(ag_news_model_flow, criterion_flow, optimizer_flow, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_flow', BVA22, BTA22)

optimizer_flow = torch.optim.Adam(ag_news_model_flow.parameters(), lr=1e-4)
ag_news_model_flow, BVA22, BTA22 = DownStreamTask(ag_news_model_flow, criterion_flow, optimizer_flow, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_flow', BVA22, BTA22)

optimizer_flow = torch.optim.Adam(ag_news_model_flow.parameters(), lr=1e-5)
ag_news_model_flow, BVA22, BTA22 = DownStreamTask(ag_news_model_flow, criterion_flow, optimizer_flow, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_flow', BVA22, BTA22)

optimizer_flow = torch.optim.Adam(ag_news_model_flow.parameters(), lr=1e-6)
ag_news_model_flow, BVA22, BTA22 = DownStreamTask(ag_news_model_flow, criterion_flow, optimizer_flow, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_flow', BVA22, BTA22)



optimizer_froz = torch.optim.Adam(ag_news_model_froz.parameters(), lr=1e-2)
ag_news_model_froz, BVA23, BTA23 = DownStreamTask(ag_news_model_froz, criterion_froz, optimizer_froz, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_froz', BVA23, BTA23)

optimizer_froz = torch.optim.Adam(ag_news_model_froz.parameters(), lr=1e-3)
ag_news_model_froz, BVA23, BTA23 = DownStreamTask(ag_news_model_froz, criterion_froz, optimizer_froz, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_froz', BVA23, BTA23)

optimizer_froz = torch.optim.Adam(ag_news_model_froz.parameters(), lr=1e-4)
ag_news_model_froz, BVA23, BTA23 = DownStreamTask(ag_news_model_froz, criterion_froz, optimizer_froz, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_froz', BVA23, BTA23)

optimizer_froz = torch.optim.Adam(ag_news_model_froz.parameters(), lr=1e-5)
ag_news_model_froz, BVA23, BTA23 = DownStreamTask(ag_news_model_froz, criterion_froz, optimizer_froz, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_froz', BVA23, BTA23)

optimizer_froz = torch.optim.Adam(ag_news_model_froz.parameters(), lr=1e-6)
ag_news_model_froz, BVA23, BTA23 = DownStreamTask(ag_news_model_froz, criterion_froz, optimizer_froz, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_froz', BVA23, BTA23)



optimizer_LNNF = torch.optim.Adam(ag_news_model_LNNF.parameters(), lr=1e-2)
ag_news_model_LNNF, BVA24, BTA24 = DownStreamTask(ag_news_model_LNNF, criterion_LNNF, optimizer_LNNF, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_LNNF', BVA24, BTA24)

optimizer_LNNF = torch.optim.Adam(ag_news_model_LNNF.parameters(), lr=1e-3)
ag_news_model_LNNF, BVA24, BTA24 = DownStreamTask(ag_news_model_LNNF, criterion_LNNF, optimizer_LNNF, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_LNNF', BVA24, BTA24)

optimizer_LNNF = torch.optim.Adam(ag_news_model_LNNF.parameters(), lr=1e-4)
ag_news_model_LNNF, BVA24, BTA24 = DownStreamTask(ag_news_model_LNNF, criterion_LNNF, optimizer_LNNF, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_LNNF', BVA24, BTA24)

optimizer_LNNF = torch.optim.Adam(ag_news_model_LNNF.parameters(), lr=1e-5)
ag_news_model_LNNF, BVA24, BTA24 = DownStreamTask(ag_news_model_LNNF, criterion_LNNF, optimizer_LNNF, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_LNNF', BVA24, BTA24)

optimizer_LNNF = torch.optim.Adam(ag_news_model_LNNF.parameters(), lr=1e-6)
ag_news_model_LNNF, BVA24, BTA24 = DownStreamTask(ag_news_model_LNNF, criterion_LNNF, optimizer_LNNF, train_loader, valid_loader, test_loader, num_epochs, 'ag_news_model_LNNF', BVA24, BTA24)



