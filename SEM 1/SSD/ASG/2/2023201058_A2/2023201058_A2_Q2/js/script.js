// Replace with your own Spotify API credentials
const clientId = 'ac911f705ae642e080949e0f65032703';
const clientSecret = 'fe58167d39d140f3b3a0ec24ee82dff1'; 

// Function to fetch Spotify API token
async function getAccessToken() {
    const response = await fetch('https://accounts.spotify.com/api/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + btoa(clientId + ':' + clientSecret),
        },
        body: 'grant_type=client_credentials'
    });

    const data = await response.json();
    console.log(data.access_token);
    return data.access_token;
}

// Your existing code

// Function to fetch data from the Spotify API and populate the dropdown
async function populateDropdown() {
    const accessToken = await getAccessToken();
    try {
        const response = await fetch('https://api.spotify.com/v1/markets', {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer ' + accessToken // Replace with your Spotify access token
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const markets = data.markets;

        const dropdown = document.getElementById('country');

        markets.forEach((market) => {
            const option = document.createElement('option');
            option.text = market;
            option.value = market;
            dropdown.appendChild(option);
        });
    } catch (error) {
        console.error('Error:', error);
    }
}

// Call the function to populate the dropdown when the page loads
populateDropdown();

// Function to fetch new releases from Spotify based on user input
async function fetchNewReleasesByInput() {
    const country = document.getElementById('country').value;
    const limit = document.getElementById('limit').value;
    const offset = document.getElementById('offset').value;

    const accessToken = await getAccessToken();

    const response = await fetch(`https://api.spotify.com/v1/browse/new-releases?country=${country}&limit=${limit}&offset=${offset}`, {
        headers: {
            'Authorization': 'Bearer ' + accessToken
        }
    });

    const data = await response.json();
    return data.albums.items;
}

// Function to handle the "Fetch New Releases" button click
function handleFetchButtonClick() {
    displayNewReleasesByInput();
}

// Attach an event listener to the "Fetch New Releases" button
const fetchButton = document.getElementById('fetchButton');
fetchButton.addEventListener('click', handleFetchButtonClick);

// Function to display new releases on the webpage based on user input
async function displayNewReleasesByInput() {
    const newReleases = await fetchNewReleasesByInput();
    const newReleasesContainer = document.getElementById('newReleases');

    // Clear existing content
    newReleasesContainer.innerHTML = '';

    newReleases.forEach(album => {
        const albumElement = document.createElement('div');
        albumElement.className = 'album-container';
        albumElement.innerHTML = `<img src="${album.images[1].url}" alt=""><h2 class="album-title"><a href="${album.external_urls.spotify}" target="_blank">${album.name}</a></h2>`;
        albumElement.innerHTML += `by <span class="artist-name"><a href="${album.artists[0].external_urls.spotify}" target="_blank">${album.artists[0].name}</a></span>`;
        for(let i = 1; i < album.artists.length; i++){
            albumElement.innerHTML += `<span class="artist-name">, <a href="${album.artists[i].external_urls.spotify}" target="_blank">${album.artists[i].name}</a></span>`;
        }
        // <p class="artist-name"><a href="${album.artists.external_urls.spotify}">${album.artists.artist.name}</a></p>`;
        newReleasesContainer.appendChild(albumElement);
    });
}
