import{ BrowserRouter, Routes, Route} from 'react-router-dom';
import Signin from './pages/Signin';
import Signup from './pages/Signup';
import Profile from './pages/profile';
import Home from './pages/Home';
import About from './pages/about';
import Header from './Component/Header';

export default function App() {
    return <BrowserRouter>
    <Header />
    <Routes>
      <Route path ="/" element={<Home />}/>
      <Route path ="/sign-in" element={<Signin />}/>
      <Route path ="/sign-up" element={<Signup />}/>
      <Route path ="/Profile" element={<Profile />}/>
      <Route path ="/About" element={<About/>}/>

    </Routes>
      </BrowserRouter>
}
