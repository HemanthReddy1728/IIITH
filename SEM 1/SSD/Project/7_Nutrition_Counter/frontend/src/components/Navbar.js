import { Link } from "react-router-dom";
import { useLogout } from "../hooks/useLogout";
import { useAuthContext } from "../hooks/useAuthContext";

const Navbar = () => {
    const { logout } = useLogout();
    const { user } = useAuthContext();

    const handleClick = () => {
        logout();
    };

    return (
        <header>
            <div className="custom-container">
                <Link to="/">
                    <div class="inline-container">
                        <div class="image-container" style={{ width: '70px', height: '50px' }}></div>
                        <h1>Nutrition Counter</h1>

                    </div>
                </Link>
                <nav>
                    {user && (
                        <div>
                            <span style={{ fontSize: '15px' }}><b>{user.name}</b></span>
                            <button onClick={handleClick}>Log out</button>
                        </div>
                    )}
                </nav>
            </div>
        </header>
    );
};

export default Navbar;