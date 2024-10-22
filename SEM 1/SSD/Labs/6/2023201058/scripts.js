document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('teamRegistrationForm');
    const teamNameInput = document.getElementById('teamName');
    const teamCoachInput = document.getElementById('teamCoach');
    const teamEmailInput = document.getElementById('teamEmail');
    const teamPasswordInput = document.getElementById('teamPassword');
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const teamCaptainSelect = document.getElementById('teamCaptain');
    const teamMembersList = document.getElementById('teamMembers');
    const passwordMatchError = document.getElementById('passwordMatchError');
    const teamMembersError = document.getElementById('teamMembersError');
    const emailError = document.getElementById('emailError');
    const submitButton = document.querySelector('button[type="submit"]');
    const darkModeToggle = document.getElementById('darkModeToggle'); // Add an element with id 'darkModeToggle' for manual dark mode toggle

    teamNameInput.addEventListener('input', validateTeamName);
    teamEmailInput.addEventListener('input', validateEmail);
    confirmPasswordInput.addEventListener('input', validatePasswordMatch);

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        if (validateForm()) {

            const teamName = teamNameInput.value;
            const teamCoach = teamCoachInput.value;
            const teamEmail = teamEmailInput.value;
            const teamCaptain = teamCaptainSelect.value;
            const teamMembers = getTeamMembers();

            // Construct the alert message
            const alertMessage = `
            Team Name: ${teamName}
            Team Coach Name: ${teamCoach}
            Team Email ID: ${teamEmail}
            Team Captain: ${teamCaptain}
            Team Members: ${teamMembers}
            `;

            const modal = document.getElementById('myModal');
            const modalMessage = document.getElementById('modalMessage');
            modalMessage.textContent = alertMessage;
            modal.style.display = 'block';

            const closeButton = document.getElementsByClassName('close')[0];
            closeButton.onclick = function () {
                modal.style.display = 'none';
            };

            form.reset();
        }
    });


    function validateTeamName() {
        const teamName = teamNameInput.value;
        const regex = /^(?=.*[A-Z])(?=.*\d)/;

        if (regex.test(teamName)) {
            teamNameInput.classList.remove('input-error');
            document.getElementById('teamNameError').textContent = '';
            return true;
        } else {
            teamNameInput.classList.add('input-error');
            document.getElementById('teamNameError').textContent = 'Invalid Username';
            return false;
        }
    }

    function validateEmail() {
        const teamEmail = teamEmailInput.value;
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

        if (emailPattern.test(teamEmail)) {
            teamEmailInput.classList.remove('input-error');
            emailError.textContent = '';
            return true;
        } else {
            teamEmailInput.classList.add('input-error');
            emailError.textContent = 'Invalid email format';
            return false;
        }
    }


    function togglePasswordVisibility(inputId, buttonId) {
        const passwordInput = document.getElementById(inputId);
        const toggleButton = document.getElementById(buttonId);

        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            toggleButton.textContent = 'Hide Password';
        } else {
            passwordInput.type = 'password';
            toggleButton.textContent = 'Show Password';
        }
    }


    document.getElementById('togglePasswordButtonTeamPassword').addEventListener('click', function () {
        togglePasswordVisibility('teamPassword', 'togglePasswordButtonTeamPassword');
    });

    document.getElementById('togglePasswordButtonConfirmPassword').addEventListener('click', function () {
        togglePasswordVisibility('confirmPassword', 'togglePasswordButtonConfirmPassword');
    });

    function validatePasswordMatch() {
        const password = teamPasswordInput.value;
        const confirmPassword = confirmPasswordInput.value;

        if (password === confirmPassword) {
            confirmPasswordInput.classList.remove('input-error');
            passwordMatchError.textContent = '';
            return true;
        } else {
            confirmPasswordInput.classList.add('input-error');
            passwordMatchError.textContent = 'Passwords do not match';
            return false;
        }
    }

    function validateForm() {
        const isTeamNameValid = validateTeamName();
        const isEmailValid = validateEmail();
        const isPasswordMatchValid = validatePasswordMatch();
        const isTeamMembersValid = validateTeamMembers();

        return isTeamNameValid && isEmailValid && isPasswordMatchValid && isTeamMembersValid;
    }

    function getTeamMembers() {
        const members = Array.from(teamMembersList.querySelectorAll('.draggable-item'));
        return members.map(member => member.textContent).join(', ');
    }

    function validateTeamMembers() {
        const members = getTeamMembers();
        if (members.length > 0) {
            teamMembersError.textContent = '';
            return true;
        } else {
            teamMembersError.textContent = 'Please add at least one team member';
            return false;
        }
    }

    function allowDrop(event) {
        event.preventDefault(); 
    }


    function drop(event, targetId) {
        event.preventDefault();
        const data = event.dataTransfer.getData("text");

        const memberElement = document.createElement("div");
        memberElement.className = "draggable-item";
        memberElement.textContent = data;

        const targetContainer = document.getElementById(targetId);
        targetContainer.appendChild(memberElement);

        teamMembersError.textContent = '';
    }

    function drag(event) {
        event.dataTransfer.setData("text", event.target.id);
    }


    submitButton.addEventListener('click', function () {
        form.submit(); 
    });
});

document.addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'm') {
        toggleDarkMode();
    }
});

function toggleDarkMode() {
    const body = document.body;
    body.classList.toggle('dark');

    const darkModeStylesheet = document.getElementById('darkModeStylesheet');
    if (body.classList.contains('dark')) {
        darkModeStylesheet.href = 'dark.css';
    } else {
        darkModeStylesheet.href = '';
    }
}

