document.addEventListener("DOMContentLoaded", function () {
    const searchIcon = document.getElementById("search-icon");
    const searchContainer = document.querySelector(".search-container");
    const searchInput = document.getElementById("search-input");
    const searchButton = document.getElementById("search-button");

    searchIcon.addEventListener("click", function () {
        searchContainer.style.display = "block";
        searchInput.focus();
    });

    searchButton.addEventListener("click", function () {
        const searchTerm = searchInput.value.trim();
        if (searchTerm !== "") {
            highlightText(searchTerm);
        }
    });

    function highlightText(text) {
        const regex = new RegExp(text, "gi");
        const content = document.querySelector("main").innerHTML;
        const highlightedContent = content.replace(
            regex,
            (match) => `<span class="highlight">${match}</span>`
        );
        document.querySelector("main").innerHTML = highlightedContent;
    }
});
